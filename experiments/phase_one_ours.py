"""Run the existing KVTC prompt-archive selector on a frozen chat prompt.

This is a quality adapter, not a bounded-memory long-generation serving engine.
Newly generated KV remains exact. No approximate baseline is labeled official.
"""
from pathlib import Path
import torch
from kvtc import KVTCCodec, KVTCConfig, feature_dim
from kvtc.cold_store import ColdStore
from . import model_adapter as A, calibration as C
from .benchmark_state import digest, source_identity
from .reference_heuristics import eviction_scores, eviction_mask, page_mass_scores
from .selector import scan_key_coefficients


class QueryRows:
    """Expose the original token-index lookup using only the rows the scorer reads."""
    def __init__(self, positions, rows):
        self.positions, self.rows = positions, rows

    def __getitem__(self, positions):
        indices = torch.searchsorted(self.positions, positions)
        if (indices >= len(self.positions)).any() or not torch.equal(self.positions[indices], positions):
            raise ValueError('query row was not captured')
        return self.rows[indices]


def prefill_scoring_rows(model, ids):
    n = len(ids)
    positions = torch.cat([torch.arange(0, n, 16), torch.arange(max(0, n-64), n)]).unique(sorted=True)
    buffers = [[] for _ in model.model.layers]
    offsets = [0 for _ in buffers]
    hooks = []
    def capture(index, output):
        start, end = offsets[index], offsets[index]+output.shape[1]
        selected = positions[(positions >= start) & (positions < end)]-start
        buffers[index].append(output[0].index_select(0, selected.to(output.device)).detach().cpu())
        offsets[index] = end
    try:
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.self_attn.q_proj.register_forward_hook(lambda m, inputs, output, i=i: capture(i, output)))
        layers, _ = A.prefill(model, ids)
    finally:
        for hook in hooks:
            hook.remove()
    return layers, [QueryRows(positions, torch.cat(rows)) for rows in buffers]


class PromptArchive:
    def __init__(self, model, manifest, directory):
        self.model = model
        self.manifest = manifest
        self.directory = Path(directory)
        self.codec = None

    def calibrate(self):
        if self.codec is not None:
            return
        # The manifest is the protocol authority. This also covers the separate
        # dual-T4 profile, which deliberately uses a smaller calibration rank.
        rank = self.manifest.get('codec_rank_cap', 1024)
        cfg = KVTCConfig(target_cr=16, pca_rank_cap=min(rank, feature_dim(self.model)),
                         svd_method='randomized', seed=42, dp_stride=1)
        self.codec = KVTCCodec(cfg, device=next(self.model.parameters()).device)
        signature = digest([C.identity(self.manifest['model'], self.manifest['revision'], cfg,
                                      self.manifest['calibration']), source_identity()])
        path = self.directory / (signature + '.pt')
        if path.exists():
            C.load(self.codec, path, signature)
            return
        keys, values = [], []
        for ids in self.manifest['calibration']:
            layers, _ = A.prefill(self.model, torch.tensor(ids), 1024)
            k, v = A.to_features(self.model, layers)
            keys.append(k)
            values.append(v)
            del layers
        self.codec.calibrate(keys, values, verbose=True)
        C.save(self.codec, path, signature)

    @torch.no_grad()
    def prefill(self, token_ids, settings):
        # The same chat prompt as every baseline. The suffix supplies the query
        # prepass, preserving all original absolute positions.
        suffix_length = min(64, len(token_ids) - 1)
        doc, query = token_ids[:-suffix_length], torch.tensor(token_ids[-suffix_length:])
        n = len(doc)
        total_budget = settings['sink'] + settings['recent'] + settings['budget']
        if n <= total_budget:
            layers, _ = A.prefill(self.model, torch.tensor(doc))
            result, _ = A.question_forward(self.model, layers, n, query)
            return result, dict(prompt_archive_used=False, retained_prompt_tokens=n,
                                generated_cache_policy='retain_all_generated_kv')
        self.calibrate()
        device = next(self.model.parameters()).device
        layers, queries = prefill_scoring_rows(self.model, torch.tensor(doc))
        heavy, _ = eviction_scores(self.model, layers, queries, n, device, stride=16, win=min(64, n))
        del queries
        hot_mask = eviction_mask(heavy, n, min(n, settings['sink'] + settings['recent']))
        hot_mask[:4] = 1
        hot_mask[max(0, n-settings['recent']):] = 1
        hot_ids = hot_mask.nonzero().flatten()
        hot_layers = A.slice_layers(layers, hot_ids)
        k, v = A.to_features(self.model, layers)
        del layers
        archive = ColdStore.encode(self.codec, k, v, settings['page_size'])
        del k, v
        coefficients, _ = scan_key_coefficients(archive, 256)
        prepass, qs = A.question_forward(self.model, hot_layers, n, query, collect_queries=True)
        del prepass
        scores = page_mass_scores(self.model, self.codec, coefficients, n, qs, settings['page_size'],
                                  256, hot_mask, device, (0, self.model.config.num_hidden_layers))
        pages = scores.argsort(descending=True)[:settings['budget']//settings['page_size']].tolist()
        active = A.recover(self.model, archive, pages, hot_layers, hot_ids)
        result, _ = A.question_forward(self.model, active.layers, n, query)
        return result, dict(prompt_archive_used=True, retained_prompt_tokens=len(active.positions),
                            archive_bytes=archive.nbytes(), selected_pages=pages,
                            generated_cache_policy='retain_all_generated_kv',
                            note='One query scan of compressed prompt; hot/page overlap means token budgets are not byte matched.')

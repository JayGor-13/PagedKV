"""Fetch pinned upstream code only; CUDA builds belong in separate GPU environments."""
import argparse
import json
from pathlib import Path
import subprocess

from experiments.benchmark_state import ROOT


def git(path, *args):
    # Trust only this explicitly selected workspace checkout, including copies
    # transferred by a different OS account. Do not change global Git config.
    return subprocess.check_output(['git', '-c', f'safe.directory={Path(path).resolve().as_posix()}',
                                    '-C', str(path), *args], text=True).strip()


def verify(name):
    spec = json.loads((ROOT / 'configs/upstream_baselines.json').read_text())[name]
    path = ROOT / spec['directory']
    if git(path, 'rev-parse', 'HEAD') != spec['commit']:
        raise ValueError(f'{name}: upstream commit differs from lock file')
    if git(path, 'status', '--porcelain', '--untracked-files=no'):
        raise ValueError(f'{name}: tracked upstream source is modified')
    submodules = git(path, 'submodule', 'status', '--recursive') if spec.get('submodules_required', True) else 'not used by this accuracy adapter'
    if spec.get('submodules_required', True) and any(line and line[0] in '-+U' for line in submodules.splitlines()):
        raise ValueError(f'{name}: initialize submodules at their pinned revisions')
    return dict(**spec, submodules=submodules)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods', nargs='+', choices=('arkvale', 'kivi', 'freekv', 'factory', 'rocketkv'), default=['arkvale', 'kivi'])
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    specs = json.loads((ROOT / 'configs/upstream_baselines.json').read_text())
    for name in args.methods:
        spec = specs[name]
        dest = ROOT / spec['directory']
        if not args.verify_only:
            if not dest.exists():
                subprocess.run(['git', 'clone', '--no-checkout', spec['url'], str(dest)], check=True)
                git(dest, 'checkout', '--detach', spec['commit'])
            elif git(dest, 'rev-parse', 'HEAD') != spec['commit']:
                raise ValueError(f'{dest} is at a different commit; use a fresh checkout')
            if spec.get('submodules_required', True):
                git(dest, 'submodule', 'update', '--init', '--recursive')
        print(json.dumps(verify(name), indent=2))


if __name__ == '__main__':
    main()

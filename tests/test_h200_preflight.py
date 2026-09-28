from scripts.h200_preflight import evaluate_cuda_toolkit, evaluate_gpu_inventory, evaluate_host_capacity


def statuses(checks):
    return {item['name']: item['status'] for item in checks}


def test_h200_inventory_requires_count_model_memory_capability_and_bf16():
    good = dict(index=0, name='NVIDIA H200', total_memory_gib=140.0,
                compute_capability=[9, 0], bf16_supported=True)
    assert set(statuses(evaluate_gpu_inventory([good] * 5, 5, 130)).values()) == {'pass'}
    bad = [dict(good, index=0, name='NVIDIA A100', total_memory_gib=80,
                compute_capability=[8, 0], bf16_supported=False)]
    result = statuses(evaluate_gpu_inventory(bad, 1, 130))
    assert result == {'gpu_count': 'pass', 'gpu_model': 'fail', 'gpu_memory': 'fail',
                      'compute_capability': 'fail', 'bf16': 'fail'}
    assert statuses(evaluate_gpu_inventory([], 5, 130))['gpu_count'] == 'fail'


def test_host_capacity_blocks_disk_but_only_warns_for_ram():
    result = statuses(evaluate_host_capacity(200, 256, 350, 512))
    assert result == {'free_disk': 'fail', 'host_ram': 'warn'}


def test_cuda_toolkit_rejects_cuda_13_and_marks_unpinned_cuda_12():
    assert evaluate_cuda_toolkit('Cuda compilation tools, release 13.1, V13.1.80')['status'] == 'fail'
    assert evaluate_cuda_toolkit('Cuda compilation tools, release 12.4, V12.4.131')['status'] == 'pass'
    assert evaluate_cuda_toolkit('Cuda compilation tools, release 12.6, V12.6.20')['status'] == 'warn'
    assert evaluate_cuda_toolkit('not nvcc output')['status'] == 'fail'

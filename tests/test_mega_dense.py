from romatch.benchmarks import MegadepthDenseBenchmark
import numpy as np

def test_mega_dense(model, name):
    mega_dense_benchmark = MegadepthDenseBenchmark("data/megadepth", h = 560, w = 560)
    mega_dense_results = mega_dense_benchmark.benchmark(model)
    return mega_dense_results

if __name__ == "__main__":
    from romatch import roma_outdoor
    device = "cuda"
    model = roma_outdoor(device = device, coarse_res = 560, use_custom_corr=True, symmetric = False, upsample_preds = False)
    experiment_name = "roma_latest"
    results = test_mega_dense(model, experiment_name)
    print(results)
    # gotten on 3.12 env with torch 2.8.0
    reference_results = {'epe': 1.581197752074192, 'mega_pck_1': 0.8516846923828125, 'mega_pck_3': 0.9566336059570313, 'mega_pck_5': 0.9714825439453125}
    assert np.isclose(results['epe'], reference_results['epe'], atol=1e-1)
    assert np.isclose(results['mega_pck_1'], reference_results['mega_pck_1'], atol=2e-1 / 100)
    assert np.isclose(results['mega_pck_3'], reference_results['mega_pck_3'], atol=2e-1 / 100)
    assert np.isclose(results['mega_pck_5'], reference_results['mega_pck_5'], atol=2e-1 / 100)
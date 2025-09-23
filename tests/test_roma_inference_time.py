import json
import numpy as np
from romatch import roma_outdoor
from romatch.benchmarks import MegadepthDenseBenchmark


def test_mega_dense(model, name):
    megadense_benchmark = MegadepthDenseBenchmark("data/megadepth", num_samples=2000, h = 560, w = 560)
    megadense_results = megadense_benchmark.benchmark(model)
    print(megadense_results)
    return megadense_results
    # json.dump(megadense_results, open(f"results/mega_dense_{name}.json", "w"))


if __name__ == "__main__":
    device = "cuda"
    model = roma_outdoor(
        device=device,
        coarse_res=560,
        upsample_res=None,
        symmetric=False,
        upsample_preds=False,
        use_custom_corr=True,
    )
    experiment_name = "roma_latest"
    results = test_mega_dense(model, experiment_name)
    reference_scores = {'epe': 1.4764922824679216, 'mega_pck_1': 0.8525611572265624, 'mega_pck_3': 0.9566116943359375, 'mega_pck_5': 0.970957763671875}
    assert np.isclose(results['epe'], reference_scores['epe'], atol=1e-1)
    assert np.isclose(results['mega_pck_1'], reference_scores['mega_pck_1'], atol=5e-1 / 100)
    assert np.isclose(results['mega_pck_3'], reference_scores['mega_pck_3'], atol=5e-1 / 100)
    assert np.isclose(results['mega_pck_5'], reference_scores['mega_pck_5'], atol=5e-1 / 100)
    # test_mega1500_poselib(model, experiment_name)

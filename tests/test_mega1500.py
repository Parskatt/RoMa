import json
from romatch.benchmarks import MegaDepthPoseEstimationBenchmark
import numpy as np

def test_mega1500(model, name):
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark("data/megadepth")
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    return mega1500_results

if __name__ == "__main__":
    from romatch import roma_outdoor
    device = "cuda"
    model = roma_outdoor(device = device, coarse_res = 672, upsample_res = 1344, use_custom_corr=True)
    experiment_name = "roma_latest"
    results = test_mega1500(model, experiment_name)[experiment_name]
    print(results)
    # gotten on 3.12 env with torch 2.8.0 
    reference_scores = [0.6235757679569996, 0.7648007367330985, 0.8630483724961098]
    assert np.isclose(results[0], reference_scores[0], atol=3e-1)
    assert np.isclose(results[1], reference_scores[1], atol=2e-1)
    assert np.isclose(results[2], reference_scores[2], atol=1e-1)
    

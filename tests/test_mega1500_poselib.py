from romatch.benchmarks import Mega1500PoseLibBenchmark

def test_mega1500_poselib(model, name):
    mega1500_benchmark = Mega1500PoseLibBenchmark("data/megadepth")
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    return mega1500_results

if __name__ == "__main__":
    from romatch import roma_outdoor
    device = "cuda"
    model = roma_outdoor(device = device, coarse_res = 672, upsample_res = 1344, use_custom_corr=True)
    experiment_name = "roma_latest"
    results = test_mega1500_poselib(model, experiment_name)
    print(results)

import json
from romatch import roma_outdoor
from romatch.benchmarks import MegadepthDenseBenchmark


def test_mega_dense(model, name):
    megadense_benchmark = MegadepthDenseBenchmark("data/megadepth", num_samples=1000, h = 560, w = 560)
    megadense_results = megadense_benchmark.benchmark(model)
    print(megadense_results)
    json.dump(megadense_results, open(f"results/mega_dense_{name}.json", "w"))


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
    test_mega_dense(model, experiment_name)
    # test_mega1500_poselib(model, experiment_name)

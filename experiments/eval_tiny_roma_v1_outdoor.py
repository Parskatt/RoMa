import torch
import os
from pathlib import Path
import json
from romatch.benchmarks import ScanNetBenchmark
from romatch.benchmarks import Mega1500PoseLibBenchmark, ScanNetPoselibBenchmark
from romatch.benchmarks import MegaDepthPoseEstimationBenchmark

def test_mega_8_scenes(model, name):
    mega_8_scenes_benchmark = MegaDepthPoseEstimationBenchmark("data/megadepth",
                                                scene_names=['mega_8_scenes_0019_0.1_0.3.npz',
                                                    'mega_8_scenes_0025_0.1_0.3.npz',
                                                    'mega_8_scenes_0021_0.1_0.3.npz',
                                                    'mega_8_scenes_0008_0.1_0.3.npz',
                                                    'mega_8_scenes_0032_0.1_0.3.npz',
                                                    'mega_8_scenes_1589_0.1_0.3.npz',
                                                    'mega_8_scenes_0063_0.1_0.3.npz',
                                                    'mega_8_scenes_0024_0.1_0.3.npz',
                                                    'mega_8_scenes_0019_0.3_0.5.npz',
                                                    'mega_8_scenes_0025_0.3_0.5.npz',
                                                    'mega_8_scenes_0021_0.3_0.5.npz',
                                                    'mega_8_scenes_0008_0.3_0.5.npz',
                                                    'mega_8_scenes_0032_0.3_0.5.npz',
                                                    'mega_8_scenes_1589_0.3_0.5.npz',
                                                    'mega_8_scenes_0063_0.3_0.5.npz',
                                                    'mega_8_scenes_0024_0.3_0.5.npz'])
    mega_8_scenes_results = mega_8_scenes_benchmark.benchmark(model, model_name=name)
    print(mega_8_scenes_results)
    json.dump(mega_8_scenes_results, open(f"results/mega_8_scenes_{name}.json", "w"))

def test_mega1500(model, name):
    mega1500_benchmark = MegaDepthPoseEstimationBenchmark("data/megadepth")
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    json.dump(mega1500_results, open(f"results/mega1500_{name}.json", "w"))

def test_mega1500_poselib(model, name):
    #model.exact_softmax = True
    mega1500_benchmark = Mega1500PoseLibBenchmark("data/megadepth", num_ransac_iter = 1, test_every = 1)
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    json.dump(mega1500_results, open(f"results/mega1500_poselib_{name}.json", "w"))

def test_mega_8_scenes_poselib(model, name):
    mega1500_benchmark = Mega1500PoseLibBenchmark("data/megadepth", num_ransac_iter = 1, test_every = 1,
                                                  scene_names=['mega_8_scenes_0019_0.1_0.3.npz',
                                                    'mega_8_scenes_0025_0.1_0.3.npz',
                                                    'mega_8_scenes_0021_0.1_0.3.npz',
                                                    'mega_8_scenes_0008_0.1_0.3.npz',
                                                    'mega_8_scenes_0032_0.1_0.3.npz',
                                                    'mega_8_scenes_1589_0.1_0.3.npz',
                                                    'mega_8_scenes_0063_0.1_0.3.npz',
                                                    'mega_8_scenes_0024_0.1_0.3.npz',
                                                    'mega_8_scenes_0019_0.3_0.5.npz',
                                                    'mega_8_scenes_0025_0.3_0.5.npz',
                                                    'mega_8_scenes_0021_0.3_0.5.npz',
                                                    'mega_8_scenes_0008_0.3_0.5.npz',
                                                    'mega_8_scenes_0032_0.3_0.5.npz',
                                                    'mega_8_scenes_1589_0.3_0.5.npz',
                                                    'mega_8_scenes_0063_0.3_0.5.npz',
                                                    'mega_8_scenes_0024_0.3_0.5.npz'])
    mega1500_results = mega1500_benchmark.benchmark(model, model_name=name)
    json.dump(mega1500_results, open(f"results/mega_8_scenes_poselib_{name}.json", "w"))

def test_scannet_poselib(model, name):
    scannet_benchmark = ScanNetPoselibBenchmark("data/scannet")
    scannet_results = scannet_benchmark.benchmark(model)
    json.dump(scannet_results, open(f"results/scannet_{name}.json", "w"))

def test_scannet(model, name):
    scannet_benchmark = ScanNetBenchmark("data/scannet")
    scannet_results = scannet_benchmark.benchmark(model)
    json.dump(scannet_results, open(f"results/scannet_{name}.json", "w"))

if __name__ == "__main__":
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1" # For BF16 computations
    os.environ["OMP_NUM_THREADS"] = "16"
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    from romatch import tiny_roma_v1_outdoor

    experiment_name = Path(__file__).stem
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = tiny_roma_v1_outdoor(device)
    #test_mega1500_poselib(model, experiment_name)
    test_mega_8_scenes_poselib(model, experiment_name)
 
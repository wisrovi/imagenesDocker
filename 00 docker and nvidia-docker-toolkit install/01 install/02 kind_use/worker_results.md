# Worker Results

| worker_name    | labels                          | available_gpu | result_by_pop | notes |
|----------------|---------------------------------|---------------|---------------|-------|
| kind-worker    | worker=worker1,nvidia.com/gpu.present=true | No            | Scheduled     | Pod scheduled correctly on GPU worker |
| kind-worker2   | worker=worker2,nvidia.com/gpu.present=true | No            | Scheduled     | Pod scheduled correctly on GPU worker |
| kind-worker3   | worker=worker3,nvidia.com/gpu.present=true | No            | Scheduled     | Pod scheduled correctly on GPU worker |
| kind-worker4   | worker=worker4,node-type=CPU   | N/A           | N/A           | CPU worker, no GPU test |
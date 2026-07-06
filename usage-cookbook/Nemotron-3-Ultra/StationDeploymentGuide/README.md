# Deploying Nemotron 3 Ultra on DGX Station

This guide serves **NVIDIA Nemotron 3 Ultra** on a single GB300-based
[DGX Station](https://www.nvidia.com/en-us/products/workstations/dgx-station/)
with [vLLM](https://docs.vllm.ai/) 0.22.0. The recommended container image is
`vllm/vllm-openai:v0.22.0`. The result is an OpenAI-compatible API on
port 8000.

Unlike the [four-node DGX Spark deployment](../SparkDeploymentGuide/README.md),
this configuration uses one GPU and offloads selected Mixture-of-Experts (MoE)
weights into the DGX Station's coherent CPU memory.

## Prerequisites

- A GB300-based DGX Station running its current NVIDIA AI software stack. The
  system combines GPU HBM and CPU LPDDR5X in a coherent memory space over
  NVLink-C2C; see the
  [DGX Station Development Guide](https://docs.nvidia.com/dgx/dgx-station-development-guide/Intro.html).
- vLLM 0.22.0. The recommended image is `vllm/vllm-openai:v0.22.0`;
  it supports `--cpu-offload-params`, `--kernel_config`, and the
  `flashinfer-trtllm` NVFP4 GEMM backend used below.
- Access to
  [`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4)
  on Hugging Face and enough local storage for the model weights.

If the model download requires authentication, log in before starting vLLM:

```shell
hf auth login
```

## Launch vLLM

Run the following command on the DGX Station:

```shell
VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1 \
VLLM_NVFP4_GEMM_BACKEND=flashinfer-trtllm \
vllm serve nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 \
  --served-model-name nemotron-ultra \
  --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --cpu-offload-gb 150 \
  --cpu-offload-params experts \
  --kernel_config '{"enable_flashinfer_autotune": false}' \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.9
```

The initial model download and load can take several minutes. Keep this process
running while clients use the API.

### Configuration notes

| Setting | Purpose |
| :--- | :--- |
| `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY=1` | Avoids consuming GPU memory for pinned offload buffers on the coherent-memory system. |
| `VLLM_NVFP4_GEMM_BACKEND=flashinfer-trtllm` | Selects the FlashInfer TensorRT-LLM backend for NVFP4 matrix multiplication. |
| `--tensor-parallel-size 1` | Runs the model on the DGX Station's single integrated Blackwell Ultra GPU. |
| `--cpu-offload-gb 150` | Makes up to 150 GiB of CPU memory available for weight offloading. |
| `--cpu-offload-params experts` | Restricts offloading to parameters whose name contains the exact `experts` segment. |
| `--kernel_config ...` | Disables FlashInfer autotuning for this launch configuration. |
| `--max-num-seqs 256` | Allows the scheduler to batch up to 256 sequences, subject to available cache memory. |
| `--gpu-memory-utilization 0.9` | Lets vLLM use up to 90 percent of GPU memory for model execution and cache. |

## Verify the API

In another terminal, confirm that the served model is available:

```shell
curl http://localhost:8000/v1/models
```

Then send a chat completion request:

```shell
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-ultra",
    "messages": [
      {"role": "user", "content": "Explain why selective expert offloading works well for a mixture-of-experts model."}
    ],
    "max_tokens": 256
  }'
```

Clients should use `nemotron-ultra` as the model name and
`http://<station-host>:8000/v1` as the OpenAI-compatible base URL. Restrict
network access to the service if it is reachable beyond the local machine.

## Troubleshooting

- **vLLM rejects an option or backend:** update the DGX Station software stack
  and use the recommended `vllm/vllm-openai:v0.22.0` image, which supports the
  offload and kernel settings used above.
- **The model download is denied:** accept the model terms on Hugging Face and
  authenticate with `hf auth login`.
- **The model runs out of memory while loading:** stop other GPU workloads and
  confirm that both offload environment variables are set on the same command
  that starts vLLM. If necessary, increase `--cpu-offload-gb` within available
  system memory or reduce `--gpu-memory-utilization`.
- **The API reports an unknown model:** send `nemotron-ultra`, the value passed
  to `--served-model-name`, rather than the Hugging Face repository name.

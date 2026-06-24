# Device Setup — running this project on any GPU / CPU / cloud

torch and torchvision are intentionally NOT in `[project.dependencies]`. You pick
the right build for your hardware with a uv extra. Everything else (timm,
torchxrayvision, sklearn, pydicom, ...) installs the same on every device.

One GPU build, `cu128`, covers BOTH an RTX 4050 (Ada, sm_89) and an RTX 5070
(Blackwell, sm_120) — newer CUDA wheels are backward-compatible with older
architectures. Only drop to `cu126` if the host driver is too old for CUDA 12.8.

## Pick your install

| Hardware / host | Command |
|---|---|
| RTX 4050, RTX 5070, most recent NVIDIA (driver supports CUDA 12.8) | `uv sync --extra cu128` |
| Older NVIDIA driver (CUDA 12.6-class) | `uv sync --extra cu126` |
| No GPU / CI / Windows or Linux CPU | `uv sync --extra cpu` |
| macOS (Apple Silicon, MPS) | `uv pip install torch torchvision` (PyPI default build) |
| RunPod / Vast / any cloud, zero-config | `uv pip install torch torchvision --torch-backend=auto` then `uv sync --inexact` |

Notes:
- You MUST pass an extra. A bare `uv sync` installs no torch (the package will not import).
- The three extras are mutually exclusive (`conflicts` in pyproject); switching is just
  re-running `uv sync --extra <other>` — uv swaps the torch build in place.
- `--torch-backend=auto` only works with `uv pip`, not `uv sync`. It probes the CUDA
  driver / AMD / Intel GPU and picks the best index, falling back to CPU. Force one with
  `--torch-backend=cu128`. Good for throwaway cloud pods where you don't want to edit files.

## RunPod recipe (reproducible)

1. Start a pod from a CUDA template (e.g. PyTorch 2.x / CUDA 12.8, Ubuntu 22.04).
2. `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. `git clone <repo> && cd research-cxr`
4. `uv sync --extra cu128`   (or `uv pip install torch torchvision --torch-backend=auto`)
5. `uv run python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"`

Check the GPU name and bf16 flag before a long run.

## Verify the install

```
uv run python -c "import torch,torchvision; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```
- `torch.version.cuda` should print 12.6 / 12.8 (or None for the cpu/mps build).
- `torch.cuda.is_available()` should be True on a GPU host.

## Runtime adapts to the device (no code edits needed)

Set in the experiment YAML; defaults are safe on small VRAM.

| Knob (YAML) | What it does | Tuning by device |
|---|---|---|
| `training.bf16` | request bf16 autocast | auto-disabled if the GPU lacks bf16 (handled by `device.bf16_supported`); leave true |
| `training.batch_size` | per-step batch | 6 GB (4050): 8-16 @224px; 12 GB (5070): 32-48; lower if OOM |
| `training.grad_accum_steps` | accumulate to fake a bigger batch | raise on small VRAM to keep effective batch constant |
| `data.loading` (`lazy`/`ram`/`gpu`) | where decoded images are cached | auto-downgrades to `lazy` when the cache won't fit free VRAM/RAM (`device.resolve_loading`) |
| `data.num_workers` | dataloader processes | ~CPU cores; 0 on Windows if you hit spawn issues |

`src/util/device.py` already: picks the device (`pick_device`), reports
GPU/VRAM/RAM (`device_info`), downgrades the cache mode to fit memory
(`resolve_loading`), and gates bf16 to capable GPUs (`bf16_supported`). So the
same config file runs on a 6 GB laptop GPU, a 12 GB card, a CPU box, or a cloud
A100 — only `batch_size` is worth hand-tuning for speed.

## Version facts (why these pins)

- `cu124` caps at torch 2.6 and predates Blackwell — removed.
- `cu126`: torch 2.6 / 2.7.   `cu128`: torch 2.7+ with Blackwell sm_120 kernels.
- We pin `torch>=2.7.0` / `torchvision>=0.22.0` so one constraint resolves on cpu, cu126, and cu128.

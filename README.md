# CUDA GEMM: matrix multiplication across the GPU memory hierarchy

Three CUDA implementations of single-precision matrix multiplication (C = A·B), each using one more level of the GPU memory hierarchy than the last, benchmarked against cuBLAS on two NVIDIA architectures.

| Kernel | Memory used | Idea |
|---|---|---|
| `gemm_naive.cu` | Global memory only | One thread per output element |
| `gemm_tiled.cu` | Shared memory | Blocks load tiles of A and B once and reuse them |
| `gemm_register.cu` | Shared memory + registers | Each thread computes an 8×8 sub-block with accumulators held in registers (64×64 block tile, BK = 8) |

Every run checks the result against `cublasSgemm` and a CPU computation, times the kernel with CUDA events, and reports GFLOPS as `2·M·K·N / seconds / 1e9`.

## Results

Tesla T4 (Turing, Google Colab), square matrices, N = 2048:

| Kernel | GFLOPS |
|---|---|
| Naive | 439 |
| Tiled | 697 |
| Register blocking | 3337 |
| cuBLAS (reference) | 5598 |

- The naive kernel stays flat at roughly 300–445 GFLOPS regardless of N: it is bound by global-memory bandwidth.
- Register blocking is the slowest kernel at N = 128 (133 GFLOPS), because 64×64 tiles leave too few blocks to occupy the GPU, and the fastest from N ≥ 1024, reaching about 7.6× the naive kernel at N = 2048.
- The same code on an RTX 3070 Ti (Ampere) shows that the best tile configuration depends on the architecture.

The full report covers rectangular shapes, a sweep of the inner dimension K, sizes that are not multiples of the tile, run-to-run repeatability, and N up to 9000.

![GFLOPS vs N](docs/figures/fig1_gflops_vs_N.png)

## Build and run

Requires the CUDA toolkit and cuBLAS. Use `-arch=sm_75` for a T4 and `-arch=sm_86` for an RTX 30-series card.

```bash
cd src
nvcc -O3 -arch=sm_75 gemm_register.cu -o reg -lcublas

./reg                     # default square sizes (256 … 4096)
./reg 512 1024 2048       # custom square sizes
./reg -r 1024 512 2048    # rectangular: A is M×K, B is K×N
```

`gemm_naive.cu` and `gemm_tiled.cu` build and run the same way.

## Repository layout

```
src/                    CUDA sources (naive, tiled, register blocking)
scripts/gemm_plots.py   Regenerates the figures from the measured data
docs/report.el.md       Full report: algorithms, methodology, results (in Greek)
docs/architecture-notes.el.md   GPU architecture background (in Greek)
docs/figures/           Plots used in the report
```

## Context

Coursework for the Modern Parallel Architectures and High-Performance Programming course, Department of Informatics and Telematics, Harokopio University of Athens (2025–2026).

# Mini-Attention

Building attention mechanisms from research papers — from high level PyTorch to low level GPU kernels.

This repo is a learning + systems implementation project focused on understanding how modern attention works mathematically and how it is optimized in real production environments.

---

## Scope

Implement attention variants across three layers:

1. PyTorch — baseline reference implementations  
2. Python DSL — Triton / Helion kernel prototypes  
3. CUDA — optimized low level kernels  

Goal is to bridge:

Paper → Math → Implementation → Kernel Optimization

---

## Implemented So Far

### PyTorch
- Multi Head Attention (MHA)
- Grouped Query Attention (GQA)
- einops based MHA
- PyTorch SDPA (FlashAttention backend reference)

### CUDA
- Mini Flash Attention (tiled + fused softmax, performance oriented)

---

## Planned Implementations

- Multi Query Attention (MQA)
- FlashAttention v1 / v2 / v3
- Sliding Window Attention
- PagedAttention (vLLM style KV cache)
- xFormers memory efficient attention
- Ring / Distributed attention
- Long context sparse attention variants

Each implemented across PyTorch → DSL → CUDA where applicable.

---

## Repository Structure

pytorch/ Reference implementations
dsl/ Triton / Helion kernels
cuda/ CUDA kernels
papers/ Reading notes and derivations
benchmarks/ Performance comparisons
docs/ Kernel walkthroughs

---

## Hardware Branches
main Vendor agnostic reference
nvidia-b200 NVIDIA optimized kernels
amd-mi300 ROCm / AMD adaptations


Branching is used to study vendor specific kernel differences similar to production ML infra repos.

---

## Project Goals

- Reproduce attention from original papers
- Understand IO and memory bottlenecks
- Study tiling, fusion, and kernel design
- Benchmark performance vs PyTorch baselines
- Learn vendor specific GPU optimizations

---

## Status

Active development.

Implementations prioritize clarity first, then performance optimization.

---

## Notes

This is a private research and learning repository.  
Code and kernels are being built incrementally alongside paper reading and benchmarking.


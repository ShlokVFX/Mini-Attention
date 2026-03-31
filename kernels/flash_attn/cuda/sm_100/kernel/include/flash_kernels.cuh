// This file is auto-generated in "gen_kernel_instantiations.py".
// Simplified variant: kernel type is KernelConfig<params...> instead of
// StaticForwardKernelConfig<FlashForwardKernelConfig{params...}>.

#pragma once

#include <map>

#include "flash_attention.cuh"
#include "forward_kernel.cuh"
#include "forward_kernel_wgmma.cuh"
#include "forward_kernel_tcgen05.cuh"

namespace flash {

typedef void (*forward_kernel_fn)(const ForwardKernelArgs);

std::map<FlashForwardKernelConfig, forward_kernel_fn>
    forward_kernels = {
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, false>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, true>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, false>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, true>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, false>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, true>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, false>>},
        // (FP16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, true>>},
        // (FP16, 128, 64, 64, 4): async+load_0_0_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, false, 0, 0, 0, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, false, false, 0, 0, 0, false, false>>},
        // (FP16, 128, 64, 64, 4): async+swizzled+load_0_0_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, false, true, 0, 0, 0, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, false, true, 0, 0, 0, false, false>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, false>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, true>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, false>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, true>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, false>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, true>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, false>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, true>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, false>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, true>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, false>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, true>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, false>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, true>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, false>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, true>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 2, 2, 2, false, false>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 2, 2, 2, true, false>>},
        // (FP16, 128, 64, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 2, 2, 2, true, true>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, false>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, true>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, false>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, true>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, false>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, true>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, false>>},
        // (FP16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, true>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, false>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, true>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, false>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, true>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, false>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, true>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, false>>},
        // (FP16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, true>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, false>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, false, true>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, false>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 0, true, true>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, false>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, false, true>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, false>>},
        // (BF16, 128, 64, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 32, 4, true, true, true, 2, 2, 2, true, true>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, false>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, false, true>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, false>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 0, true, true>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, false>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, false, true>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, false>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_0_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 0, 2, true, true>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, false>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, false, true>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, false>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 0, true, true>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, false>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, true>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, false>>},
        // (BF16, 128, 64, 64, 4): async+eager+swizzled+load_0_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, true, true>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, false>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, false, true>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, false>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 0, true, true>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, false>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, false, true>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, false>>},
        // (BF16, 128, 128, 32, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 32, 4, true, true, true, 2, 2, 2, true, true>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, false>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, false, true>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, false>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_0_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 0, true, true>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, false>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, false, true>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, false>>},
        // (BF16, 128, 128, 64, 4): async+eager+swizzled+load_2_2_2_tiles+buffer+opt_softmax
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, true}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 64, 4, true, true, true, 2, 2, 2, true, true>>},

        // =====================================================================
        // SM120 (Blackwell RTX 5090) — Bc=128 large-tile configurations
        //
        // Why Bc=128?  Blackwell has 100 KB smem/SM vs 64 KB on SM86.
        //   smem(Br=64,  Bc=128) = (64+256) * 128 * 2 = 81,920 B  < 100 KB ✓
        //   smem(Br=128, Bc=128) = (128+256)* 128 * 2 = 98,304 B  < 100 KB ✓
        //
        // Arithmetic intensity improvement vs Bc=64:
        //   Bc=64:  ~16 FLOP/byte  (Br=Bc=64, d=128)
        //   Bc=128: ~21 FLOP/byte  (+33%)
        //
        // This translates directly to higher GPU utilization at long seq lengths.
        // =====================================================================

        // ---- FP16, Br=64, Bc=128, NW=4 ---- smem = 81,920 B ----- //
        // Whole-RF Q, streaming K+V (0,2,2) — best for register-heavy workloads
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 128, 4, true, true, true, 0, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 128, 4, true, true, true, 0, 2, 2, false, false>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 128, 4, true, true, true, 0, 2, 2, false, true},  &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 128, 4, true, true, true, 0, 2, 2, false, true>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 128, 4, true, true, true, 0, 2, 2, true,  false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 128, 4, true, true, true, 0, 2, 2, true,  false>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 128, 4, true, true, true, 0, 2, 2, true,  true},  &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 128, 4, true, true, true, 0, 2, 2, true,  true>>},
        // Streaming all (2,2,2)
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 128, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 128, 4, true, true, true, 2, 2, 2, false, false>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 128, 4, true, true, true, 2, 2, 2, false, true},  &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 128, 4, true, true, true, 2, 2, 2, false, true>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 128, 4, true, true, true, 2, 2, 2, true,  false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 128, 4, true, true, true, 2, 2, 2, true,  false>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 128, 4, true, true, true, 2, 2, 2, true,  true},  &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 128, 4, true, true, true, 2, 2, 2, true,  true>>},

        // ---- FP16, Br=64, Bc=128, NW=8 ---- smem = 81,920 B ----- //
        // 8 warps: better latency hiding on Blackwell's wider warp scheduler
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 128, 8, true, true, true, 0, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 128, 8, true, true, true, 0, 2, 2, false, false>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 128, 8, true, true, true, 0, 2, 2, false, true},  &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 128, 8, true, true, true, 0, 2, 2, false, true>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 128, 8, true, true, true, 0, 2, 2, true,  false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 128, 8, true, true, true, 0, 2, 2, true,  false>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 128, 8, true, true, true, 0, 2, 2, true,  true},  &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 128, 8, true, true, true, 0, 2, 2, true,  true>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 128, 8, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 128, 8, true, true, true, 2, 2, 2, false, false>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 128, 8, true, true, true, 2, 2, 2, false, true},  &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 128, 8, true, true, true, 2, 2, 2, false, true>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 128, 8, true, true, true, 2, 2, 2, true,  false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 128, 8, true, true, true, 2, 2, 2, true,  false>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 128, 8, true, true, true, 2, 2, 2, true,  true},  &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 128, 8, true, true, true, 2, 2, 2, true,  true>>},

        // ---- FP16, Br=128, Bc=128, NW=4 ---- smem = 98,304 B ---- //
        // Maximum tile size: highest arithmetic intensity; fits in Blackwell's 100 KB smem
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 128, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 128, 4, true, true, true, 2, 2, 2, false, false>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 128, 4, true, true, true, 2, 2, 2, false, true},  &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 128, 4, true, true, true, 2, 2, 2, false, true>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 128, 4, true, true, true, 2, 2, 2, true,  false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 128, 4, true, true, true, 2, 2, 2, true,  false>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 128, 4, true, true, true, 2, 2, 2, true,  true},  &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 128, 4, true, true, true, 2, 2, 2, true,  true>>},

        // ---- FP16, Br=128, Bc=128, NW=8 ---- smem = 98,304 B ---- //
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 128, 8, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 128, 8, true, true, true, 2, 2, 2, false, false>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 128, 8, true, true, true, 2, 2, 2, false, true},  &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 128, 8, true, true, true, 2, 2, 2, false, true>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 128, 8, true, true, true, 2, 2, 2, true,  false}, &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 128, 8, true, true, true, 2, 2, 2, true,  false>>},
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 128, 8, true, true, true, 2, 2, 2, true,  true},  &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 128, 8, true, true, true, 2, 2, 2, true,  true>>},

        // ---- BF16, Br=64, Bc=128, NW=4 ---- smem = 81,920 B ---- //
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 128, 4, true, true, true, 0, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 128, 4, true, true, true, 0, 2, 2, false, false>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 128, 4, true, true, true, 0, 2, 2, false, true},  &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 128, 4, true, true, true, 0, 2, 2, false, true>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 128, 4, true, true, true, 0, 2, 2, true,  false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 128, 4, true, true, true, 0, 2, 2, true,  false>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 128, 4, true, true, true, 0, 2, 2, true,  true},  &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 128, 4, true, true, true, 0, 2, 2, true,  true>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 128, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 128, 4, true, true, true, 2, 2, 2, false, false>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 128, 4, true, true, true, 2, 2, 2, false, true},  &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 128, 4, true, true, true, 2, 2, 2, false, true>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 128, 4, true, true, true, 2, 2, 2, true,  false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 128, 4, true, true, true, 2, 2, 2, true,  false>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 128, 4, true, true, true, 2, 2, 2, true,  true},  &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 128, 4, true, true, true, 2, 2, 2, true,  true>>},

        // ---- BF16, Br=64, Bc=128, NW=8 ---- smem = 81,920 B ---- //
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 128, 8, true, true, true, 0, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 128, 8, true, true, true, 0, 2, 2, false, false>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 128, 8, true, true, true, 0, 2, 2, false, true},  &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 128, 8, true, true, true, 0, 2, 2, false, true>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 128, 8, true, true, true, 0, 2, 2, true,  false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 128, 8, true, true, true, 0, 2, 2, true,  false>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 128, 8, true, true, true, 0, 2, 2, true,  true},  &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 128, 8, true, true, true, 0, 2, 2, true,  true>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 128, 8, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 128, 8, true, true, true, 2, 2, 2, false, false>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 128, 8, true, true, true, 2, 2, 2, false, true},  &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 128, 8, true, true, true, 2, 2, 2, false, true>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 128, 8, true, true, true, 2, 2, 2, true,  false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 128, 8, true, true, true, 2, 2, 2, true,  false>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 128, 8, true, true, true, 2, 2, 2, true,  true},  &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 128, 8, true, true, true, 2, 2, 2, true,  true>>},

        // ---- BF16, Br=128, Bc=128, NW=4 ---- smem = 98,304 B --- //
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 128, 4, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 128, 4, true, true, true, 2, 2, 2, false, false>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 128, 4, true, true, true, 2, 2, 2, false, true},  &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 128, 4, true, true, true, 2, 2, 2, false, true>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 128, 4, true, true, true, 2, 2, 2, true,  false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 128, 4, true, true, true, 2, 2, 2, true,  false>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 128, 4, true, true, true, 2, 2, 2, true,  true},  &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 128, 4, true, true, true, 2, 2, 2, true,  true>>},

        // ---- BF16, Br=128, Bc=128, NW=8 ---- smem = 98,304 B --- //
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 128, 8, true, true, true, 2, 2, 2, false, false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 128, 8, true, true, true, 2, 2, 2, false, false>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 128, 8, true, true, true, 2, 2, 2, false, true},  &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 128, 8, true, true, true, 2, 2, 2, false, true>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 128, 8, true, true, true, 2, 2, 2, true,  false}, &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 128, 8, true, true, true, 2, 2, 2, true,  false>>},

        // ====================================================================
        // B200 (SM_100) EXCLUSIVE — Bc=256 large-tile configs
        //
        // B200 (GB200 Blackwell datacenter) provides up to 228 KB of
        // addressable shared memory per CTA — more than double SM_120's ~100 KB.
        // Increasing Bc to 256 raises arithmetic intensity by ~50% over Bc=128:
        //   Bc=128: AI ≈ 21.3 FLOP/byte
        //   Bc=256: AI ≈ 32.0 FLOP/byte  (Br=Bc=256 asymptote ≈ 64)
        //
        // smem usage:  (Br + 2*Bc) * 128 * 2 bytes
        //   Br=64,  Bc=256 → 147,456 B = 144 KB  < 228 KB ✓
        //   Br=128, Bc=256 → 163,840 B = 160 KB  < 228 KB ✓
        // ====================================================================

        // ---- FP16, Br=64, Bc=256, NW=8 ---- smem = 147,456 B (144 KB) ---- //
        // B2_1: whole-RF Q, stream K (0,0,4) — V streaming dominant
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 256, 8, true, true, true, 0, 0, 4, false, true},  &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 256, 8, true, true, true, 0, 0, 4, false, true>>},
        // B2_2: whole-RF Q, stream K+V (0,2,4)
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 256, 8, true, true, true, 0, 2, 4, false, true},  &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 64, 256, 8, true, true, true, 0, 2, 4, false, true>>},

        // ---- FP16, Br=128, Bc=256, NW=8 ---- smem = 163,840 B (160 KB) ---- //
        // B2_3: whole-RF Q, stream K+V (0,2,2)
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 256, 8, true, true, true, 0, 2, 2, false, true},  &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 256, 8, true, true, true, 0, 2, 2, false, true>>},
        // B2_4: stream all (2,2,2) + double-buffer — maximum pipeline depth
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 256, 8, true, true, true, 2, 2, 2, true,  true},  &flash_forward_kernel<KernelConfig<torch::kFloat16, 128, 128, 256, 8, true, true, true, 2, 2, 2, true,  true>>},

        // ---- BF16, Br=64, Bc=256, NW=8 ---- smem = 147,456 B (144 KB) ---- //
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 256, 8, true, true, true, 0, 0, 4, false, true},  &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 256, 8, true, true, true, 0, 0, 4, false, true>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 64, 256, 8, true, true, true, 0, 2, 4, false, true},  &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 64, 256, 8, true, true, true, 0, 2, 4, false, true>>},

        // ---- BF16, Br=128, Bc=256, NW=8 ---- smem = 163,840 B (160 KB) ---- //
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 256, 8, true, true, true, 0, 2, 2, false, true},  &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 256, 8, true, true, true, 0, 2, 2, false, true>>},
        {FlashForwardKernelConfig{torch::kBFloat16, 128, 128, 256, 8, true, true, true, 2, 2, 2, true,  true},  &flash_forward_kernel<KernelConfig<torch::kBFloat16, 128, 128, 256, 8, true, true, true, 2, 2, 2, true,  true>>}
};

// ===========================================================================
// WGMMA kernel map — B200 / SM_100 exclusive
//
// These variants replace the QK GEMM inner loop with wgmma.mma_async,
// reading Q and K directly from smem via a descriptor (no ldmatrix).
// The PV GEMM still uses mma.sync since P is computed in registers.
//
// Requirements:
//   - FP16 only (wgmma_qk_loop has a static_assert for half)
//   - n_warps must be a multiple of 4 (one warpgroup = 4 warps)
//   - Bc must be 64 (wgmma.m64n64k16) or 128 (wgmma.m64n128k16)
//   - Br / (n_warps/4) == 64  so each warpgroup handles exactly 64 Q rows
//
// Configs instantiated:
//   KW_1: Br=64, Bc=64,  NW=4 — 1 WG, S-flat=32, m64n64k16
//   KW_2: Br=64, Bc=128, NW=4 — 1 WG, S-flat=64, m64n128k16
//   KW_3: Br=128,Bc=64,  NW=8 — 2 WGs, S-flat=32
//   KW_4: Br=128,Bc=128, NW=8 — 2 WGs, S-flat=64  (max tile for WGMMA path)
// ===========================================================================

// Helper: verify wgmma constraint at compile time
template <typename KC>
constexpr void check_wgmma_config() {
    static_assert(KC::n_warps % WARPS_PER_WARPGROUP == 0,
                  "WGMMA: n_warps must be a multiple of 4");
    static_assert((KC::Br / (KC::n_warps / WARPS_PER_WARPGROUP)) == WGMMA_M,
                  "WGMMA: each warpgroup must handle exactly 64 Q rows");
    static_assert(KC::Bc == 64 || KC::Bc == 128,
                  "WGMMA: Bc must be 64 or 128");
}

std::map<FlashForwardKernelConfig, forward_kernel_fn>
    wgmma_forward_kernels = {
        // ---- KW_1: Br=64, Bc=64, NW=4 ---- smem = 49,152 B ---- //
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, true}, &flash_forward_kernel_wgmma<KernelConfig<torch::kFloat16, 128, 64, 64, 4, true, true, true, 0, 2, 2, false, true>>},

        // ---- KW_2: Br=64, Bc=128, NW=4 ---- smem = 81,920 B ---- //
        {FlashForwardKernelConfig{torch::kFloat16, 128, 64, 128, 4, true, true, true, 0, 2, 2, false, true}, &flash_forward_kernel_wgmma<KernelConfig<torch::kFloat16, 128, 64, 128, 4, true, true, true, 0, 2, 2, false, true>>},

        // ---- KW_3: Br=128, Bc=64, NW=8 ---- smem = 65,536 B ---- //
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 64, 8, true, true, true, 0, 2, 2, false, true}, &flash_forward_kernel_wgmma<KernelConfig<torch::kFloat16, 128, 128, 64, 8, true, true, true, 0, 2, 2, false, true>>},

        // ---- KW_4: Br=128, Bc=128, NW=8 ---- smem = 98,304 B ---- //
        {FlashForwardKernelConfig{torch::kFloat16, 128, 128, 128, 8, true, true, true, 0, 2, 2, false, true}, &flash_forward_kernel_wgmma<KernelConfig<torch::kFloat16, 128, 128, 128, 8, true, true, true, 0, 2, 2, false, true>>},
};


// ===========================================================================
// tcgen05 kernel map — B200 / SM_100a exclusive
//
// A lightweight key struct for the tcgen05 dispatch table.  Unlike the mma.sync
// and WGMMA maps (keyed on FlashForwardKernelConfig with 13 fields), the tcgen05
// kernel is fixed at BLK_M=BLK_N=HEAD_D=64 and only varies on dtype and causal.
//
// Extend this when HEAD_D=128/256 variants are added.
// ===========================================================================

struct Tcgen05KernelConfig {
    torch::ScalarType dtype;
    int               head_dim;
    bool              is_causal;

    bool operator<(const Tcgen05KernelConfig &o) const {
        if (dtype    != o.dtype)    return dtype    < o.dtype;
        if (head_dim != o.head_dim) return head_dim < o.head_dim;
        return is_causal < o.is_causal;
    }
};

typedef void (*tcgen05_kernel_fn)(const ForwardKernelArgs);

std::map<Tcgen05KernelConfig, tcgen05_kernel_fn>
    tcgen05_forward_kernels = {
        // FP16, HEAD_D=64, non-causal  (the only validated config for now)
        {Tcgen05KernelConfig{torch::kFloat16, 64, false},
         &flash_forward_kernel_tcgen05},
};

} // namespace flash

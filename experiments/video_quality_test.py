"""
Video quality comparison: CogVideoX-2b with standard attention vs SageAttention.

Usage:
    # Standard attention (baseline):
    python experiments/video_quality_test.py --mode standard --prompt "a cat walking in snow"

    # SageAttention (real thu-ml):
    python experiments/video_quality_test.py --mode sage --prompt "a cat walking in snow"

    # Both back-to-back (saves two videos for visual comparison):
    python experiments/video_quality_test.py --mode both --prompt "a cat walking in snow"

Output: experiments/output/video_<mode>_<seed>.mp4

Model: CogVideoX-2b (2B params, ~8GB fp16, DiT architecture)
       Generates 6s video at 720x480, 8fps, 50 steps by default.
       Reduce --frames and --steps for faster first-pass tests.
"""

import argparse
import time
import os
import torch

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def patch_cogvideox_attention(pipe, use_sage: bool):
    """Swap CogVideoX transformer attention to use sageattn() or revert to SDPA."""
    from diffusers.models.attention_processor import Attention

    if use_sage:
        from sageattention import sageattn
        from diffusers.models.embeddings import apply_rotary_emb

        class CogVideoXSageAttnProcessor:
            """
            Mirrors CogVideoXAttnProcessor2_0 but replaces SDPA with sageattn().
            CogVideoX concatenates text + video tokens for joint self-attention,
            then splits the output back. The processor must return (video, text) tuple.
            """
            def __call__(
                self,
                attn: Attention,
                hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                attention_mask=None,
                image_rotary_emb=None,
            ):
                text_seq_length = encoder_hidden_states.size(1)
                # Concatenate text + video tokens
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

                B, N, _ = hidden_states.shape
                h = attn.heads
                head_dim = hidden_states.shape[-1] // h  # will recompute from qkv

                q = attn.to_q(hidden_states)
                k = attn.to_k(hidden_states)
                v = attn.to_v(hidden_states)

                inner_dim = k.shape[-1]
                head_dim = inner_dim // h

                # (B, N, H*D) -> (B, H, N, D)
                q = q.view(B, N, h, head_dim).transpose(1, 2)
                k = k.view(B, N, h, head_dim).transpose(1, 2)
                v = v.view(B, N, h, head_dim).transpose(1, 2)

                if attn.norm_q is not None:
                    q = attn.norm_q(q)
                if attn.norm_k is not None:
                    k = attn.norm_k(k)

                # Apply RoPE to video portion only
                if image_rotary_emb is not None:
                    q[:, :, text_seq_length:] = apply_rotary_emb(q[:, :, text_seq_length:], image_rotary_emb)
                    if not attn.is_cross_attention:
                        k[:, :, text_seq_length:] = apply_rotary_emb(k[:, :, text_seq_length:], image_rotary_emb)

                # sageattn needs fp16/bf16; contiguous last dim
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()
                orig_dtype = q.dtype
                if orig_dtype not in (torch.float16, torch.bfloat16):
                    q, k, v = q.half(), k.half(), v.half()

                out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
                out = out.to(orig_dtype)

                # (B, H, N, D) -> (B, N, H*D)
                out = out.transpose(1, 2).reshape(B, N, h * head_dim)
                out = attn.to_out[0](out)
                out = attn.to_out[1](out)

                # Split back into text and video
                encoder_out, hidden_out = out.split(
                    [text_seq_length, N - text_seq_length], dim=1
                )
                return hidden_out, encoder_out

        processor = CogVideoXSageAttnProcessor()
        label = "SageAttention (INT8 QK, FP16 V)"
    else:
        from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0
        processor = CogVideoXAttnProcessor2_0()
        label = "Standard SDPA (CogVideoX)"

    # apply to all transformer blocks
    count = 0
    for module in pipe.transformer.modules():
        if isinstance(module, Attention):
            module.set_processor(processor)
            count += 1

    print(f"Patched {count} attention modules -> {label}")
    return count


def run_inference(pipe, prompt, seed, num_frames, num_steps, guidance_scale):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    t0 = time.perf_counter()
    result = pipe(
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return result.frames[0], elapsed


def save_video(frames, path, fps=8):
    import imageio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)
    print(f"Saved: {path}  ({len(frames)} frames @ {fps}fps)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["standard", "sage", "both"], default="both")
    parser.add_argument("--prompt", default="a golden retriever playing fetch on a beach, sunny day, cinematic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=25,
                        help="Number of video frames (25=~3s at 8fps, 49=~6s)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--model", default="THUDM/CogVideoX-2b",
                        help="HF model ID. CogVideoX-2b ~8GB fp16")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    from diffusers import CogVideoXPipeline

    pipe = CogVideoXPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.enable_model_cpu_offload()
    print(f"Model loaded. GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    modes = ["standard", "sage"] if args.mode == "both" else [args.mode]

    for mode in modes:
        use_sage = mode == "sage"
        patch_cogvideox_attention(pipe, use_sage)

        print(f"\n[{mode.upper()}] Generating: \"{args.prompt}\"")
        print(f"  frames={args.frames}  steps={args.steps}  seed={args.seed}")

        frames, elapsed = run_inference(
            pipe, args.prompt, args.seed,
            args.frames, args.steps, args.guidance
        )

        out_path = os.path.join(OUTPUT_DIR, f"video_{mode}_seed{args.seed}.mp4")
        save_video(frames, out_path)

        mem_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Time: {elapsed:.1f}s  Peak VRAM: {mem_gb:.1f}GB")
        torch.cuda.reset_peak_memory_stats()

    if args.mode == "both":
        print(f"\nOpen both videos to compare quality:")
        print(f"  Standard: {OUTPUT_DIR}/video_standard_seed{args.seed}.mp4")
        print(f"  Sage:     {OUTPUT_DIR}/video_sage_seed{args.seed}.mp4")
        print(f"  Expected: visually identical, sage should be faster")


if __name__ == "__main__":
    main()

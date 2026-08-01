# Model & Quantization Guidance

GB10's unified memory + LPDDR5X bandwidth (~273 GB/s) changes which models make sense
compared to a typical discrete-GPU setup. This is the most important doc in this repo for
getting good performance rather than just "getting it running."

## The core tradeoff

| Model type | Behavior on GB10                                                        |
| ------------ | ---------------------------------------------------------------------------|
| Dense, unquantized (BF16/FP16) | Bandwidth-bound. A 7B BF16 dense model tops out around ~19 tok/s theoretical — more compute doesn't help past the memory-bandwidth ceiling. |
| Dense, quantized (AWQ 4-bit etc.) | Meaningfully faster than unquantized, since fewer bytes move per token. Recommended default for dense models. |
| MoE (Mixture of Experts) | Best fit for GB10. Only active expert(s) need to be "hot" per token — e.g. a ~40GB MoE model might move only ~3B params worth of weights per token, giving quality competitive with much larger dense models at a fraction of the effective bandwidth cost. |

## Practical guidance

- **If you have to run a dense model, quantize it.** AWQ 4-bit or similar is close to a
  requirement for acceptable throughput on anything above ~3B parameters.
- **Prefer MoE architectures when available for your use case.** With 128GB unified
  memory, you can fit a large MoE (e.g. 40-80GB total) while only paying the bandwidth
  cost of its active experts per token — this is where GB10 genuinely shines relative to
  its raw bandwidth number.
- **NVFP4 is worth checking for newer models.** NVIDIA's Blackwell-generation low-precision
  format enables larger models to fit in unified memory — some community reports note
  loading up to 200B-parameter NVFP4 models on a single Spark depending on architecture
  and runtime config. Check whether your target model has an NVFP4 release before
  defaulting to AWQ.
- **Don't assume "fits in 128GB" is the only constraint.** KV cache scales with context
  length and concurrent requests — leave real headroom rather than sizing a model to use
  the full 128GB for weights alone.

## Known-good starting points

| Model                          | Format       | Notes                                     |
| ---------------------------------- | -------------- | ---------------------------------------------|
| TinyLlama-1.1B                     | Q8_0 GGUF     | Good first smoke test for llama.cpp build validation |
| 7B-class dense models (Llama, Qwen) | AWQ 4-bit     | Reasonable throughput; avoid BF16/FP16 unless quality-critical |
| Qwen3-Coder-Next (MoE, ~40GB total, ~3B active) | native/AWQ | Strong throughput-to-quality tradeoff on GB10 |
| 80B-class MoE models                | Quantized     | Slower per-token than smaller dense models, but competitive quality at higher effective capacity |

## Backend selection (vLLM)

`FLASH_ATTN` is the stable, recommended attention backend on GB10 as of this writing.
`FLASHINFER` is listed as an option in vLLM but has a known bug in community GB10 builds
(a `non_blocking=None TypeError` on warmup) — stick with `FLASH_ATTN` until this is
resolved upstream. See [`docs/vllm-notes.md`](vllm-notes.md) for more backend-specific
detail.

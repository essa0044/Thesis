# VLM test, run 1 (13 Oct 2026): what happened

**Setup:** qwen2.5vl:3b (Q4_K_M, 3.5 GB loaded) in Ollama 0.33.3 on an i5-12500H laptop with an RTX 3050 Laptop GPU (4 GB). 32 COCO photos, half with the target.

**Result: the pipeline works, but the model broke.**
- The first 3 answers were valid JSON and correct (`{"found": false}` on 3 photos without a person).
- From the 4th photo on, every answer was `@@@@@@...` until the length cap. Only 3 of 32 answers were usable (19% on the person set, 0% on the dog set).
- Latency: about 25-28 s per image. `ollama ps` showed the model split 57% CPU / 43% GPU, because it does not fit in the 4 GB GPU.

**Most likely cause (not proven):** the model not fitting in the GPU and being split across CPU and GPU. The odd token counts Ollama reported (0-1 output tokens for 80-character answers) suggest the model runner itself was in a bad state, not that the prompt was wrong.

**Run 2 tests that:**
1. qwen3-vl:2b (1.9 GB), which fits entirely in the 4 GB GPU.
2. qwen2.5vl:3b again, on the CPU only. If its answers are fine there, the GPU split was the problem.

Raw answers are now saved in each result JSON ("samples"), so failures can be read afterwards.

**For the thesis:** this laptop is a development machine, not the target. On a Jetson Orin Nano (8 GB shared memory) a 3B model fits completely, so these latencies are not Jetson numbers.

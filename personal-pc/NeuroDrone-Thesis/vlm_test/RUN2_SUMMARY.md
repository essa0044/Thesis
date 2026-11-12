# VLM test, run 2 (13 Oct 2026)

Laptop: i5-12500H, RTX 3050 Laptop GPU (4 GB), Ollama 0.33.3. 32 COCO photos (16 "the person", 16 "the dog"; half contain the target). Results are in `results\run_2026-10-13_1558`.

| Model | Where it ran | Valid answers | Found / not found right | Left-right error of the target centre | Time per new image |
|---|---|---|---|---|---|
| Qwen3-VL 2B | 82% GPU / 18% CPU | 32 / 32 | 32 / 32 | 0.4% of image width (max 2.5%) | ~13-17 s |
| Qwen2.5-VL 3B | CPU only | 32 / 32 | 32 / 32 | 7.8% of image width (one 43% miss: it chose a teddy bear over a small dog) | ~38-44 s |
| Qwen2.5-VL 3B (run 1) | 57% CPU / 43% GPU | 3 / 32 | 3 / 3 | - | ~25-28 s |

**What this shows**
- The whole chain works with real models: image -> VLM -> JSON -> target bearing.
- Run 1's `@@@@` answers came from the CPU/GPU split, not from the model or our code: the same model on the CPU alone answered everything correctly.
- Qwen3-VL 2B is both smaller and far more precise at locating the target. It is the first candidate for the Jetson.

**Read the times carefully**
- Some answers took only ~1.5 s. Those photos had already been sent once (the negatives are shared between the two sets), so Ollama reused its cache. On a drone every frame is new, so ~15 s is the honest laptop figure.
- That gap means generating the answer is fast. Most of the time goes into reading the image (~1,180 input tokens). A smaller image size should cut the time sharply: test this in experiment 5.1.
- These are laptop numbers. The Jetson's numbers have to be measured on the Jetson.

**Bug found and fixed**
Both models give boxes on a 0-1000 scale. The parser guessed the scale from the numbers, so a box whose values were all below ~470 was read as pixels (one photo per model). There is now a `coords` setting (`norm1000` for Qwen models in Ollama) in the config, the benchmark (`--coords`) and the run script. The errors in the table above use the corrected reading. Two new tests cover it; 87/87 tests pass.

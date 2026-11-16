# Choosing the VLM {#sec:vlm}

This chapter answers RQ2: which small, locally run VLM finds a described object best, and how accuracy and latency change with model size, quantisation and image size. The benchmark measures exactly what the drone needs — one bounding box for a free-text description — through the same request and the same parser the drone uses.

## Experimental set-up

**Hardware and server.** All benchmark numbers come from one workstation: AMD Ryzen Threadripper PRO 5955WX (16 cores), 128 GB RAM and an NVIDIA RTX 6000 Ada GPU with 48 GB of memory, Windows 11. The models were served by Ollama 0.40.13 [@ollama] on its Vulkan back-end (the GPU driver could not be updated to the version Ollama's CUDA back-end needs). One request was sent at a time and the models stayed loaded, so the latencies are steady-state times per image, not loading times. Absolute latencies on the Jetson will be higher; this chapter uses them to compare models with each other.

**Datasets.** Three public data sets with their own labels were used; no self-labelled images (Table @tbl:datasets).

Table: Data sets of the VLM benchmark. {#tbl:datasets}

| Data set | Content | Split used | Scored samples |
|---|---|---|---|
| RefCOCO (UNC) [@yu2016refcoco] | COCO photographs with referring expressions, one object each | testA, testB | 5,607 and 5,045 (full); 500 per split in the side studies |
| gRefCOCO [@liu2023gres] | RefCOCO images with expressions that refer to no object or several | val | 4,187 (full) |
| RefDrone [@sun2025refdrone] | VisDrone drone photographs with referring expressions | test, single- and no-target | 1,254 targets + 152 no-target; 84 % of the targets cover less than 1 % of the image |

RefDrone is the closest public match to the drone's task: views from a drone, small objects, and expressions that sometimes describe nothing in the picture. RefCOCO and gRefCOCO are the standard grounding benchmarks on everyday photographs and allow a check against published numbers.

**Prompt, parser and metrics.** Every model received the drone prompt of Chapter @sec:design (system message: "You are the visual grounding module of a small drone. You reply with ONE JSON object and nothing else."), temperature 0. The models do not answer in the requested 0–1 coordinates (Qwen3-VL uses 0–1000, Qwen2.5-VL pixels of the image as the server resized it), so the first 50 samples of each task select each model's convention and are never scored. The main metric is the standard accuracy at intersection-over-union (IoU) ≥ 0.5 on samples whose target is present. For samples without a target, the share answered "not found" is reported: a model that draws a box for an object that is not there sends a drone to the wrong place. Latency is the median time per request. 95 % confidence intervals use the Wilson score interval; with 500 samples they are about ±3–4 points, with the full RefDrone set about ±3 points.

**Models.** Ollama's q4_K_M (4-bit) builds of the Qwen3-VL Instruct models [@bai2025qwen3vl] at 2, 4 and 8 billion parameters (B) were the main candidates; the "thinking" tags were not used. For comparison: Qwen2.5-VL-3B [@bai2025qwen25vl], the Qwen3.5 models (0.8 B–9 B), Gemma 4 (E2B, E4B, 12 B), LFM3-VL 2B, Bonsai-2, Nemotron-3.5-Bolt E4B, and two large reference models that do not fit an 8 GB computer: Qwen3-VL-30B-A3B (a mixture of experts with 3 B active parameters) and Qwen3-VL-32B.

## Model size

Table @tbl:size gives the results of the three Qwen3-VL sizes on the complete test sets, and Figure @fig:vlmsize plots drone-image accuracy against latency for all models measured.

Table: Qwen3-VL Instruct at 4 bit on the complete test sets: accuracy at IoU ≥ 0.5 (%) and median latency on the workstation. {#tbl:size}

| | 2B | 4B | 8B |
|---|---|---|---|
| RefDrone, all targets (95 % CI) | 43.9 (41–47) | **69.3 (67–72)** | 63.2 (60–66) |
| RefDrone, small targets (< 1 % of the image) | 38.5 | **65.9** | 59.4 |
| RefDrone, target absent: answered "not found" | **79.6** | 48.7 | 75.7 |
| gRefCOCO val | 83.8 | **86.9** | 84.0 |
| RefCOCO testA | 83.5 | **90.6** | 89.5 |
| RefCOCO testB | 78.1 | **83.4** | 83.1 |
| Model file | 1.9 GB | 3.3 GB | 6.2 GB |
| Median latency | 0.38 s | 0.56–0.61 s | 0.83 s |

![Accuracy on drone images (RefDrone, full test set, IoU ≥ 0.5) against median latency on the workstation GPU: Qwen3-VL Instruct at 2B, 4B and 8B in three quantisation levels, and two large reference models. Marker area grows with the model file size.](figures/fig_vlm_size.png){#fig:vlmsize width=90%}

The 4B model is the best on every task. On drone images it finds 69 % of the targets, against 63 % for the 8B and 44 % for the 2B, and it keeps this lead on the small targets that make up most of RefDrone. Larger is not better: the 8B model is 1.5 times slower and worse on drone images. A second draw of 500 samples per task with another random seed gave the same ranking (4B 68–70 %, 8B 62 %, 2B 41–42 % on RefDrone); differences of less than about 4 points between models should not be claimed.

The large reference models confirm that the limit on drone images is not model size. Qwen3-VL-30B-A3B reached 69.4 % and the dense 32B model 68.5 % on RefDrone — the same as the 4B model at 3.3 GB, with 6–7 times the memory and, for the 32B, 2.5 s per image. They are better in two respects: on RefCOCO testB (+6–7 points) and in saying "not found" when the object is absent (66–72 % against 49 %).

**The 4B model's weakness is invented objects.** When the described object is not in the drone image, Qwen3-VL-4B still draws a box in about half of the cases; the 2B and 8B models do so in about a quarter. This is the main reason for the consensus rule of Chapter 3: a goal is accepted only when two of three answers agree on the same place.

**Our numbers against the published ones.** The Qwen3-VL technical report gives a RefCOCO average (RefCOCO, RefCOCO+ and RefCOCOg, full precision, the model's own prompt) of 85.6 / 89.0 / 89.1 % for 2B / 4B / 8B [@bai2025qwen3vl]. The mean of our testA and testB results is 80.8 / 87.0 / 86.3 %: the same order, 2–5 points lower, as expected with a 4-bit model, a prompt that allows "not found", and a subset of the splits. For Qwen2.5-VL-3B the gap to its published scores [@bai2025qwen25vl] was much larger (46–59 % against 84–92 %), and came from refusals, not from wrong boxes: it answered "not found" for 33–49 % of the present targets, and 84–89 % of the boxes it did draw were correct. A prompt that allows "not found" is therefore not neutral; models differ in how willing they are to use it.

## Quantisation

The q8_0 (8-bit) and bf16 (16-bit) builds of the three Qwen3-VL models were run on the full RefDrone set and on the same 500 samples of RefCOCO and gRefCOCO (Table @tbl:quant).

Table: Quantisation of Qwen3-VL: accuracy at IoU ≥ 0.5 (%), file size and median latency. RefDrone: 1,254 samples; RefCOCO and gRefCOCO: 500 each. {#tbl:quant}

| Model | File | RefDrone | RefCOCO testA | RefCOCO testB | gRefCOCO | Latency |
|---|---|---|---|---|---|---|
| 2B q4_K_M | 1.9 GB | 43.9 | 81.2 | 76.6 | 83.7 | 0.38 s |
| 2B q8_0 | 2.7 GB | 50.9 | 83.8 | 78.6 | 83.7 | 0.40 s |
| 2B bf16 | 4.3 GB | 51.5 | 83.4 | 78.4 | 83.7 | 0.55 s |
| **4B q4_K_M** | **3.3 GB** | **69.3** | 88.8 | 80.8 | 89.3 | **0.57 s** |
| 4B q8_0 | 5.1 GB | 69.1 | 89.2 | 81.2 | 87.3 | 0.60 s |
| 4B bf16 | 8.9 GB | 69.4 | 89.4 | 80.8 | 87.7 | 1.01 s |
| 8B q4_K_M | 6.2 GB | 63.2 | 87.6 | 82.4 | 84.5 | 0.84 s |
| 8B q8_0 | 9.9 GB | 63.9 | 88.0 | 83.0 | 86.5 | 0.90 s |
| 8B bf16 | 18 GB | 63.9 | 88.0 | 82.8 | 86.1 | 1.65 s |

The 4B and 8B models lose nothing measurable at 4 bit: all differences are within one point on drone images and two points elsewhere, while the 4-bit model is 1.7–2 times faster than the 16-bit one and needs about a third of its memory. The 2B model is the exception: at 4 bit it loses 7 points on drone images (43.9 against 50.9 %), mostly by answering "not found" for targets that are present. On the Jetson, where memory bandwidth is the limit, the advantage of 4 bit is expected to be larger than on the workstation.

## Image size

Ollama resizes every image to about 1,000 image tokens before the model sees it: the prompt held about 1,200 tokens whether the longest image side was 512 or 2,048 pixels. Sending larger images therefore does not give small targets more pixels, and the latency hardly changes (0.54–0.61 s). Accuracy on RefDrone fell only when detail was lost before the up-scaling: 62 / 66 / 66 / 69 % at 448 / 640 / 896 / 1,024 px, and 65 % at 512 px; a second sample draw gave 67.2 % at 896 px against 67.4 % at 1,024 px. On RefCOCO the image size between 384 and 1,024 px made no difference. The drone sends 896 px images: as accurate as larger ones, and less work to encode and transfer on the Jetson.

## Newer models and other answer formats

**Newer and other model families.** None of the newer small models was better overall (Table @tbl:newmodels). Qwen3.5-9B found slightly more drone targets (72 %) but answered "found" for 70 % of the absent ones; the smaller Qwen3.5 models answered "found" to almost everything, which inflates their box scores. Gemma 4 drew few correct boxes (14–20 % on RefDrone), Bonsai-2 (ternary weights) and Nemotron-3.5-Bolt almost none. LFM3-VL 2B (1.1 GB) was the most interesting: 58.5 % on RefDrone at 0.24–0.32 s per image, with 85 % of absent targets answered correctly, but clearly weaker than Qwen3-VL-4B on everyday photographs.

Table: Newer models (4-bit, Ollama). RefDrone: full set; RefCOCO and gRefCOCO: 500 samples. "Absent" = share of no-target samples answered "not found". {#tbl:newmodels}

| Model | RefDrone | RefDrone absent | RefCOCO testA | gRefCOCO | Latency |
|---|---|---|---|---|---|
| Qwen3-VL-4B Instruct | 0.69 | 0.51 | 0.90 | 0.89 | 0.75 s |
| Qwen3.5-9B | 0.72 | 0.30 | 0.83 | 0.75 | 0.83 s |
| Qwen3.5-4B | 0.61 | 0.74 | 0.81 | 0.66 | 0.65 s |
| Qwen3.5-2B | 0.66 | 0.14 | 0.73 | 0.62 | 0.43 s |
| Qwen3.5-0.8B | 0.54 | 0.00 | 0.67 | 0.67 | 0.58 s |
| Gemma 4 12B | 0.20 | 0.36 | 0.50 | 0.46 | 0.85 s |
| Gemma 4 E4B | 0.18 | 0.41 | 0.53 | 0.66 | 0.52 s |
| Gemma 4 E2B | 0.14 | 0.84 | 0.38 | 0.51 | 0.32 s |
| LFM3-VL 2B | 0.59 | 0.85 | 0.85 | 0.67 | 0.24–0.32 s |

(The latency of Qwen3-VL-4B in this run, 0.75 s, is higher than in Table @tbl:size because these runs used Ollama's native API with a different request path; within a table the models are comparable.)

**Two-step strategies.** Because Ollama caps the image tokens, the only way to give a small target more pixels is to send less of the image. Four strategies were tested on RefDrone with Qwen3-VL-4B: *zoom* (ask again on a crop around the first box), *tiles* (if the full image says "not found", ask the four overlapping quarter images), *check* (ask a yes/no question about the box), and tiles plus check. Tiles gave the best drone-image result of the whole study — 73.4 % at 896 px, 0.62 s median — but lowered the correct "not found" answers on absent targets from 51 % to 29 %, because four more chances to answer also mean four more chances to invent. Zoom added 3 points at twice the latency. The yes/no check rejected half of the correct boxes. On RefCOCO none of the strategies changed the result by more than 1–2 points. With tiles, LFM3-VL 2B reached 68 % on RefDrone, close to the single-image Qwen3-VL-4B, at about half the time.

**Regions instead of boxes.** VLN on the Fly [@tayar2026vln] asks the VLM to name one of nine image regions (a 3 × 3 grid) with a confidence. On the same samples, naming a region was clearly worse than drawing a box and taking the region of its centre (Qwen3-VL-4B: 52 against 76 % on RefDrone, 57 against 88 % on gRefCOCO), and only 15–25 % faster. The stated confidence was useless as a filter: 77–96 % of the answers claimed at least 0.7, right or wrong.

**Thinking mode.** With reasoning switched on, Qwen3.5-4B and 9B became worse at grounding (51–53 % on RefDrone), a third of the answers could not be read, and one answer took 11–25 s. Thinking modes are not usable for this task.

## Decision

The drone uses **Qwen3-VL-4B-Instruct at 4 bit (q4_K_M, 3.3 GB), with 896 px images and the single-image prompt.** It is the most accurate model on drone images that fits an 8 GB computer, as accurate as models seven to eight times its size, loses nothing by 4-bit quantisation, and takes about 0.6 s per image on a desktop GPU. Its tendency to invent absent objects is handled by the consensus rule rather than by a second question, which cost more correct answers than it saved. **LFM3-VL 2B with tiles** is kept as the fast alternative, to be compared with the 4B model on the Jetson itself (stage 1 of the deployment plan, Chapter 6).

**Limits of this evaluation.** The latencies are from a desktop GPU on the Vulkan back-end, not from the Jetson. RefDrone's images are taken from greater heights than an indoor drone flies at, and its targets (cars, people) differ from the lab's objects; the found rate on real lab images is measured in stage 3 of the deployment plan. Each model was run with one prompt; prompt tuning per model might narrow some gaps, but the drone needs one prompt that works.

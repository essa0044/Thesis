NeuroDrone - first VLM test (Ollama on this PC)
================================================

What it does
- Checks that Ollama is installed and running (starts the server if needed).
- Run 2 (now the default): downloads qwen3-vl:2b (about 2 GB), which fits fully in a 4 GB GPU,
  and re-runs run 1's qwen2.5vl:3b on the CPU only, to check whether the GPU split caused run 1's garbage.
- Makes a small Python environment in .venv (numpy, pillow, pyyaml).
- Runs neurodrone\scripts\bench_vlm.py on 32 labelled photos:
  16 for "the person" and 16 for "the dog". Half of each set contain the target, half do not.
- Saves everything in a new results\run_<date_time> folder: one JSON per model and test set,
  run_log.txt, hardware.txt (CPU/GPU) and ollama_ps.txt (whether the model ran on GPU or CPU).

How to run
- Double-click run_vlm_test.bat. With an NVIDIA GPU it takes a few minutes after the download;
  on CPU only it can take 20+ minutes.
- To compare more models, run in PowerShell from this folder, for example:
  .\run_vlm_test.ps1 qwen2.5vl:3b gemma3:4b

What the numbers mean
- latency_s       seconds per answer (on this PC, not on the Jetson)
- parse_success   share of answers that were valid JSON the drone software can use
- found_accuracy  share of photos where the model was right about whether the target is there
- x_mae_norm      average left/right error of the target centre, as a fraction of image width
  (x_mae_deg assumes a 69 degree camera and means little for these photos)

Test photos: 32 images from the COCO dataset (the coco128 subset published by Ultralytics),
with COCO's own object labels. COCO images carry various Creative Commons licenses.

# NeuroDrone - first VLM test through Ollama (Windows PowerShell 5.1 or newer)
# Runs neurodrone\scripts\bench_vlm.py against the local Ollama server and saves the results.
# Start it by double-clicking run_vlm_test.bat, or in PowerShell:  .\run_vlm_test.ps1 qwen3-vl:2b-instruct qwen2.5vl:3b@cpu
# A model name ending in @cpu runs that model on the CPU only (a copy with num_gpu 0), to compare with the GPU.

$ErrorActionPreference = "Continue"
$here    = Split-Path -Parent $MyInvocation.MyCommand.Path
$root    = Split-Path -Parent $here
$repo    = Join-Path $root "neurodrone"
$results = Join-Path (Join-Path $here "results") ("run_" + (Get-Date -Format "yyyy-MM-dd_HHmm"))
New-Item -ItemType Directory -Force -Path $results | Out-Null
$log = Join-Path $results "run_log.txt"

# Run 2 defaults: a model small enough to sit fully in a 4 GB GPU, and run 1's model on the CPU only
$models = @("qwen3-vl:2b-instruct", "qwen2.5vl:3b@cpu")
if ($args.Count -gt 0) { $models = @($args) }

function Log([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line
    Add-Content -Path $log -Value $line -Encoding UTF8
}

Log ("NeuroDrone VLM test started. Models: " + ($models -join ", "))

# 1. Ollama: find it, make sure the server is running -------------------------------------------
$ollama = $null
$cmd = Get-Command ollama -ErrorAction SilentlyContinue
if ($cmd) { $ollama = $cmd.Source }
if (-not $ollama) {
    $cand = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $cand) { $ollama = $cand }
}
if (-not $ollama) {
    Log "ERROR: Ollama was not found. Install it from https://ollama.com/download and run this again."
    exit 1
}
$ver = (& $ollama --version 2>&1 | Out-String).Trim()
Log ("Ollama found: " + $ollama + " | " + $ver)

function Test-Ollama {
    try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 3 | Out-Null; return $true }
    catch { return $false }
}
if (-not (Test-Ollama)) {
    Log "The Ollama server is not running. Starting it..."
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; ($i -lt 30) -and (-not (Test-Ollama)); $i++) { Start-Sleep -Seconds 1 }
    if (-not (Test-Ollama)) { Log "ERROR: could not start the Ollama server."; exit 1 }
}
Log "Ollama server is up."

# 2. Hardware info (the GPU decides whether Ollama runs on the GPU or the CPU) -------------------
$hw = Join-Path $results "hardware.txt"
("Collected " + (Get-Date -Format o)) | Out-File -FilePath $hw -Encoding utf8
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors |
    Format-List | Out-File -FilePath $hw -Append -Encoding utf8
Get-CimInstance Win32_ComputerSystem |
    Select-Object @{ n = "RAM_GB"; e = { [math]::Round($_.TotalPhysicalMemory / 1GB, 1) } } |
    Format-List | Out-File -FilePath $hw -Append -Encoding utf8
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion |
    Format-List | Out-File -FilePath $hw -Append -Encoding utf8
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    (nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>&1 | Out-String) |
        Out-File -FilePath $hw -Append -Encoding utf8
}
Log "Hardware info saved to results\hardware.txt"

# 3. Python environment (a private .venv next to this script) ------------------------------------
$venv = Join-Path $here ".venv"
$vpy  = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $vpy)) {
    $cands = @(
        @{ exe = "py";     pre = @("-3") },
        @{ exe = "python"; pre = @() },
        @{ exe = (Join-Path $env:USERPROFILE "anaconda3\python.exe");  pre = @() },
        @{ exe = (Join-Path $env:USERPROFILE "miniconda3\python.exe"); pre = @() }
    )
    $base = $null
    foreach ($c in $cands) {
        $exe = $c.exe
        $pre = $c.pre
        try {
            $ok = & $exe @pre -c "import sys; print(sys.version_info >= (3, 9))" 2>$null
            if ("$ok".Trim() -eq "True") { $base = $c; break }
        } catch { }
    }
    if (-not $base) {
        Log "ERROR: no Python 3.9+ found. Install Python from python.org (tick 'Add python.exe to PATH') and run this again."
        exit 1
    }
    $exe = $base.exe
    $pre = $base.pre
    Log ("Creating a Python environment with: " + $exe + " " + ($pre -join " "))
    & $exe @pre -m venv $venv
    if (-not (Test-Path $vpy)) { Log "ERROR: could not create the Python environment."; exit 1 }
}
Log "Installing numpy, pillow and pyyaml into the test environment..."
& $vpy -m pip install --quiet --disable-pip-version-check numpy pillow pyyaml 2>&1 | ForEach-Object { Log "$_" }

# 4. Download the models (first time only) -----------------------------------------------------
function Pull-Model([string]$name) {
    Log "Downloading model $name (skipped if it is already on this PC)..."
    & $ollama pull $name
    return ($LASTEXITCODE -eq 0)
}
$ready = @()
foreach ($m in $models) {
    $cpu = $m.EndsWith("@cpu")
    $base = $m -replace '@cpu$', ''
    $ok = Pull-Model $base
    if ((-not $ok) -and $base.EndsWith("-instruct")) {
        $base = $base -replace '-instruct$', ''
        Log "Trying $base instead..."
        $ok = Pull-Model $base
    }
    if (-not $ok) { Log "ERROR: could not download $base - skipping it."; continue }
    $name = $base
    if ($cpu) {
        # a copy of the model that Ollama keeps entirely on the CPU
        $name = ($base -replace '[:/\\.]', '-') + "-cpu"
        $mf = Join-Path $results ("Modelfile_" + $name)
        ("FROM " + $base + "`nPARAMETER num_gpu 0`n") | Out-File -FilePath $mf -Encoding ascii
        & $ollama create $name -f $mf 2>&1 | ForEach-Object { Log "$_" }
        if ($LASTEXITCODE -ne 0) { Log "ERROR: could not make the CPU-only copy of $base - skipping it."; continue }
        Log "Made $name (CPU only)."
    }
    $ready += $name
    $safe = $name -replace '[:/\\]', '_'
    (& $ollama show $name 2>&1 | Out-String) | Out-File -FilePath (Join-Path $results ("model_" + $safe + ".txt")) -Encoding utf8
}

# 5. Benchmark: every model on every test set ----------------------------------------------------
$env:PYTHONIOENCODING = "utf-8"
$bench = Join-Path $repo "scripts\bench_vlm.py"
$sets = @(
    @{ name = "person"; instruction = "the person" },
    @{ name = "dog";    instruction = "the dog" }
)
foreach ($m in $ready) {
    $safe = $m -replace '[:/\\]', '_'
    foreach ($s in $sets) {
        $dir = Join-Path $here ("images\" + $s.name)
        $n = @(Get-ChildItem -Path $dir -Filter *.jpg).Count
        $out = Join-Path $results ("{0}__{1}.json" -f $safe, $s.name)
        $labels = Join-Path $dir "labels.json"
        Log ("Running " + $m + " on '" + $s.instruction + "' (" + $n + " images)...")
        & $vpy -u $bench --base-url "http://127.0.0.1:11434/v1" --model $m --images $dir `
            --instruction $s.instruction --labels $labels -n $n --warmup 1 --timeout 300 --max-tokens 512 --coords norm1000 `
            --out $out 2>&1 | ForEach-Object { Log "$_" }
    }
    ("--- " + $m + " ---") | Out-File -FilePath (Join-Path $results "ollama_ps.txt") -Append -Encoding utf8
    (& $ollama ps 2>&1 | Out-String) | Out-File -FilePath (Join-Path $results "ollama_ps.txt") -Append -Encoding utf8
    & $ollama stop $m 2>&1 | Out-Null   # free the GPU before the next model
}

("finished " + (Get-Date -Format o)) | Out-File -FilePath (Join-Path $results "DONE.txt") -Encoding utf8
Log ("All done. Results are in " + $results)

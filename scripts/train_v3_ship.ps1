param (
    [int]$Epochs = 13,
    [string]$LosoRunDir = "runs\v3-4signers-24-8clips-20260915-215753"
)

$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false

Set-Location "D:\Dev\Workspaces\VSLR-Workspace"

$RunId = Get-Date -Format "yyyyMMdd-HHmmss"
$RunDir = "runs\v3-ship-4signers-24-8clips-13ep-$RunId"
$RunDirPosix = $RunDir.Replace('\', '/')
New-Item -ItemType Directory -Force $RunDir | Out-Null

Write-Host "========================================================="
Write-Host "STARTING V3 SHIP CANDIDATE TRAINING ($Epochs epochs): $RunDir"
Write-Host "Target: All 768 clips P01-P04 on dataset/recordings_v3_4x24_8"
Write-Host "========================================================="

function Assert-StepSuccess($stepName) {
    if ($LASTEXITCODE -ne 0) {
        Write-Error "FAIL-FAST: Step '$stepName' failed with exit code $LASTEXITCODE!"
        exit $LASTEXITCODE
    }
}

function Assert-FilesExistAndNonEmpty($dir, $fileList) {
    foreach ($f in $fileList) {
        $p = Join-Path $dir $f
        if (-not (Test-Path $p)) {
            Write-Error "FAIL-FAST: Expected artifact missing: $p"
            exit 1
        }
        $item = Get-Item $p
        if ($item.Length -eq 0) {
            Write-Error "FAIL-FAST: Expected artifact is empty (0 bytes): $p"
            exit 1
        }
    }
}

# 0. RECORD IMMUTABLE PROVENANCE & ENVIRONMENT METADATA
Write-Host "[0/5] Recording environment metadata and invariants..."
git rev-parse HEAD | Set-Content "$RunDir\base-commit.txt"
git branch --show-current | Set-Content "$RunDir\branch.txt"
git status --porcelain | Set-Content "$RunDir\worktree-status.txt"
python -m pip freeze | Set-Content "$RunDir\pip-freeze.txt"
python --version | Set-Content "$RunDir\python-version.txt"
nvidia-smi | Set-Content "$RunDir\gpu-info.txt"

# Record old model hash invariant
$oldModelHash = (Get-FileHash -Algorithm SHA256 "models\gesture_lstm.pt").Hash.ToLower()
$oldModelHash | Set-Content "$RunDir\old-model-hash.txt"

# Record selected LOSO run path and verifier receipt
$LosoRunDir | Set-Content "$RunDir\selected-loso-run.txt"
python scripts\verify_v3_run.py $LosoRunDir | Set-Content "$RunDir\selected-loso-verifier-receipt.txt"
Assert-StepSuccess "Verify Selected LOSO Run"

# Source & script hashes
python -c "import hashlib, glob; from pathlib import Path; lines = [f'{hashlib.sha256(Path(p).read_bytes()).hexdigest().lower()}  {Path(p).as_posix()}' for p in sorted(glob.glob('src/prototype_3_gestures/**/*.py', recursive=True))]; Path('$RunDirPosix/source-hashes.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')"
python -c "import hashlib, glob; from pathlib import Path; lines = [f'{hashlib.sha256(Path(p).read_bytes()).hexdigest().lower()}  {Path(p).as_posix()}' for p in sorted(glob.glob('scripts/*.*'))]; Path('$RunDirPosix/script-hashes.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')"

Assert-FilesExistAndNonEmpty $RunDir @(
    "base-commit.txt", "branch.txt", "worktree-status.txt", "pip-freeze.txt", "python-version.txt", "gpu-info.txt",
    "old-model-hash.txt", "selected-loso-run.txt", "selected-loso-verifier-receipt.txt", "source-hashes.txt", "script-hashes.txt"
)

# 1. GENERATE AND LOCK DATASET MANIFEST
Write-Host "[1/5] Generating and locking dataset manifest..."
python scripts/build_dataset_manifest.py `
  --data-dir dataset/recordings_v3_4x24_8 `
  --recording-plan dataset/recording_plan_v3_4x24_8.json `
  --labels-file dataset/labels_v2_24.txt `
  --out-csv "$RunDir\dataset_files_sha256.csv"
Assert-StepSuccess "Generate Dataset Manifest"
Assert-FilesExistAndNonEmpty $RunDir @("dataset_files_sha256.csv")

# Record data fingerprint
python -c "from prototype_3_gestures.prepare_train import discover_clips, data_fingerprint; from pathlib import Path; clips = discover_clips('dataset/recordings_v3_4x24_8'); fp = data_fingerprint(clips); Path('$RunDirPosix/dataset-fingerprint.txt').write_text(fp + '\n', encoding='utf-8')"
Assert-FilesExistAndNonEmpty $RunDir @("dataset-fingerprint.txt")

# Copy verified loso_report.json to enable automatic three-way PAIR OK verification
Copy-Item "$LosoRunDir\loso_report.json" "$RunDir\loso_report.json" -Force

@"
Ship Training Command:
vslr-train --data-dir dataset/recordings_v3_4x24_8 --recording-plan dataset/recording_plan_v3_4x24_8.json --labels-file dataset/labels_v2_24.txt --cache-dir dataset/processed/landmark_cache_v3_4x24_8 --epochs $Epochs --augment 120 --batch-size 32 --learning-rate 0.001 --seed 42 --num-workers 4 --model-dir $RunDir
"@ | Set-Content "$RunDir\commands.txt" -Encoding utf8
Assert-FilesExistAndNonEmpty $RunDir @("commands.txt")

# 2. RUN SHIP TRAINING (768 Clips, 13 Epochs, 120 Augmentations)
Write-Host "[2/5] Running Ship Training on 768 clips for $Epochs epochs..."
vslr-train `
  --data-dir dataset/recordings_v3_4x24_8 `
  --recording-plan dataset/recording_plan_v3_4x24_8.json `
  --labels-file dataset/labels_v2_24.txt `
  --cache-dir dataset/processed/landmark_cache_v3_4x24_8 `
  --epochs $Epochs `
  --augment 120 `
  --batch-size 32 `
  --learning-rate 0.001 `
  --seed 42 `
  --num-workers 4 `
  --model-dir $RunDir 2>&1 | Tee-Object "$RunDir\ship_training.log"
Assert-StepSuccess "Ship Training Execution"
Assert-FilesExistAndNonEmpty $RunDir @("ship_training.log", "gesture_lstm.pt", "metrics.json", "labels.json")

# 3. VERIFY SHIP CANDIDATE ARTIFACTS
Write-Host "[3/5] Verifying ship candidate artifacts and three-way pairing..."
python scripts\verify_ship_candidate.py $RunDir --loso-run $LosoRunDir
Assert-StepSuccess "Ship Candidate Verification"

# 4. GENERATE RECURSIVE CHECKSUMS (UTF-8 NO BOM)
Write-Host "[4/5] Generating recursive SHA256SUMS.txt (UTF-8 no BOM)..."
python -c "import hashlib, glob; from pathlib import Path; run_dir = Path('$RunDirPosix'); lines = [];
for p in sorted(run_dir.rglob('*')):
    if p.is_file() and p.name != 'SHA256SUMS.txt':
        sha = hashlib.sha256(p.read_bytes()).hexdigest().lower()
        rel = p.relative_to(run_dir).as_posix()
        lines.append(f'{sha}  {rel}')
(run_dir / 'SHA256SUMS.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
"
Assert-FilesExistAndNonEmpty $RunDir @("SHA256SUMS.txt")

# 5. RE-VERIFY MODEL INVARIANT
Write-Host "[5/5] Re-verifying production model invariant..."
$currentProdHash = (Get-FileHash -Algorithm SHA256 "models\gesture_lstm.pt").Hash.ToLower()
if ($currentProdHash -ne $oldModelHash) {
    Write-Error "FATAL: models\gesture_lstm.pt was modified during ship training!"
    exit 1
}

Write-Host "========================================================="
Write-Host "V3 SHIP CANDIDATE SUCCESSFULLY TRAINED AND VERIFIED: $RunDir"
Write-Host "Candidate Checkpoint: $RunDir\gesture_lstm.pt"
Write-Host "========================================================="

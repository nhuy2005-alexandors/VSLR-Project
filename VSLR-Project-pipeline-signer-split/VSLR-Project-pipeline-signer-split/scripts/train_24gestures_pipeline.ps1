$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false

Set-Location "D:\Dev\Workspaces\VSLR-Workspace"

$RunId = Get-Date -Format "yyyyMMdd-HHmmss"
$RunDir = "runs\v2-4signers-24-$RunId"
New-Item -ItemType Directory -Force $RunDir | Out-Null
Write-Host "========================================================="
Write-Host "STARTING 24-GESTURE 4-SIGNER RUN: $RunDir"
Write-Host "========================================================="

# 0. RECORD ENVIRONMENT METADATA
git rev-parse HEAD | Set-Content "$RunDir\base-commit.txt"
git branch --show-current | Set-Content "$RunDir\branch.txt"
git status --porcelain | Set-Content "$RunDir\worktree-status.txt"
python -m pip freeze | Set-Content "$RunDir\pip-freeze.txt"
python --version | Set-Content "$RunDir\python-version.txt"
nvidia-smi | Set-Content "$RunDir\gpu-info.txt"
Copy-Item "dataset_files_sha256.csv" "$RunDir\dataset_files_sha256.csv" -Force

@"
LOSO Command:
vslr-train --data-dir dataset/recordings_v2_4x24 --recording-plan dataset/recording_plan_v2_4x24.json --labels-file dataset/labels_v2_24.txt --cache-dir dataset/processed/landmark_cache_v2_4x24 --loso --epochs 40 --augment 120 --batch-size 32 --learning-rate 0.001 --seed 42 --num-workers 4 --model-dir $RunDir

Ship Command:
vslr-train --data-dir dataset/recordings_v2_4x24 --recording-plan dataset/recording_plan_v2_4x24.json --labels-file dataset/labels_v2_24.txt --cache-dir dataset/processed/landmark_cache_v2_4x24 --epochs 40 --augment 120 --batch-size 32 --learning-rate 0.001 --seed 42 --num-workers 4 --model-dir $RunDir
"@ | Set-Content "$RunDir\commands.txt" -Encoding utf8

Write-Host "Running pytest before training..."
python -m pytest -q | Tee-Object "$RunDir\test.log"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pytest suite failed prior to training! Aborting."
    exit 1
}

$DataDir = "dataset\recordings_v2_4x24"
$Plan = "dataset\recording_plan_v2_4x24.json"
$Labels = "dataset\labels_v2_24.txt"
$Cache = "dataset\processed\landmark_cache_v2_4x24"

# 1. RUN VIDEO GATE
Write-Host "[1/5] Running vslr-check video gate on $DataDir..."
vslr-check --data-dir $DataDir --recording-plan $Plan --labels-file $Labels --cache-dir $Cache --min-hand-ratio 0.5
if ($LASTEXITCODE -ne 0) {
    Write-Error "Video gate vslr-check failed with exit code $LASTEXITCODE"
    exit 1
}

# 2. RUN LOSO TRAINING (4-Folds)
Write-Host "[2/5] Running LOSO 4-fold training..."
vslr-train --data-dir $DataDir --recording-plan $Plan --labels-file $Labels --cache-dir $Cache --loso `
  --epochs 40 --augment 120 --batch-size 32 --learning-rate 0.001 `
  --seed 42 --num-workers 4 --model-dir $RunDir 2>&1 |
  Tee-Object "$RunDir\loso.log"

if ($LASTEXITCODE -ne 0) {
    Write-Error "LOSO training failed with exit code $LASTEXITCODE"
    exit 1
}

# 3. RUN SHIP TRAINING (Final model on all 576 clips)
Write-Host "[3/5] Running Ship training..."
vslr-train --data-dir $DataDir --recording-plan $Plan --labels-file $Labels --cache-dir $Cache `
  --epochs 40 --augment 120 --batch-size 32 --learning-rate 0.001 `
  --seed 42 --num-workers 4 --model-dir $RunDir 2>&1 |
  Tee-Object "$RunDir\ship.log"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Ship training failed with exit code $LASTEXITCODE"
    exit 1
}

# 4. VERIFY ARTIFACT PAIRING
Write-Host "[4/5] Verifying artifact pairing contract..."
$env:RUN_DIR = (Resolve-Path $RunDir).Path
@'
import json, os, sys
from pathlib import Path
from prototype_3_gestures.prepare_train import artifact_pair_matches, file_sha256
from prototype_3_gestures.vsl3.model import load_checkpoint

root = Path(os.environ["RUN_DIR"])
report = json.loads((root / "loso_report.json").read_text(encoding="utf-8"))
metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
checkpoint = root / "gesture_lstm.pt"

assert artifact_pair_matches(report["training_signature"], metrics, checkpoint), "artifact_pair_matches failed!"
assert metrics["checkpoint_sha256"] == file_sha256(checkpoint), "checkpoint sha256 mismatch!"
model, labels, config = load_checkpoint(checkpoint)
assert len(labels) == 24, f"Expected 24 labels, got {len(labels)}"
assert config["features_version"] == 3
assert config["input_dim"] == 203
assert report["training_signature"] == metrics["training_signature"] == config["training_signature"]
print("=========================================================")
print("PAIR OK: Artifacts verified successfully!")
print("checkpoint =", checkpoint)
print("labels =", len(labels), "features_version =", config["features_version"], "input_dim =", config["input_dim"])
print("=========================================================")
'@ | python -

if ($LASTEXITCODE -ne 0) {
    Write-Error "Artifact pairing verification failed!"
    exit 1
}

# 5. FINALIZE & MACHINE AUDIT
Write-Host "[5/5] Finalizing artifacts and running machine audit..."
python scripts/finalize_run_artifacts.py --run-dir "$RunDir" --data-dir "$DataDir" --recording-plan "$Plan" --labels-file "$Labels" --cache-dir "$Cache" --expected-clips 576
if ($LASTEXITCODE -ne 0) {
    Write-Error "Finalize run artifacts failed!"
    exit 1
}

python scripts/audit_run_artifacts.py --run-dir "$RunDir" --data-dir "$DataDir" --recording-plan "$Plan" --labels-file "$Labels" --cache-dir "$Cache" --expected-clips 576
if ($LASTEXITCODE -ne 0) {
    Write-Error "Machine audit failed!"
    exit 1
}

Write-Host "========================================================="
Write-Host "ALL 15 PHASES COMPLETED SUCCESSFULLY! Artifacts in: $RunDir"
Write-Host "========================================================="

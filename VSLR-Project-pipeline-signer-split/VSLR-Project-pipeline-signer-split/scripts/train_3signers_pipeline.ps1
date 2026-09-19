$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false

Set-Location "D:\Dev\Workspaces\VSLR-Workspace"

$RunId = Get-Date -Format "yyyyMMdd-HHmmss"
$RunDir = "runs\v1-3signers-$RunId"
New-Item -ItemType Directory -Force $RunDir | Out-Null
Write-Host "========================================================="
Write-Host "STARTING OVERNIGHT 3-SIGNER RUN: $RunDir"
Write-Host "========================================================="

git rev-parse HEAD | Set-Content "$RunDir\base-commit.txt"
git status --porcelain | Set-Content "$RunDir\worktree-status.txt"

$DataDir = "dataset\recordings_v1_p123"
$Plan = "dataset\recording_plan_p123.json"

# 1. RUN LOSO TRAINING (3-Folds)
Write-Host "[1/4] Running LOSO 3-fold training..."
vslr-train --data-dir $DataDir --recording-plan $Plan --loso `
  --epochs 40 --augment 120 --batch-size 32 --learning-rate 0.001 `
  --seed 42 --num-workers 4 --model-dir $RunDir 2>&1 |
  Tee-Object "$RunDir\loso.log"

if ($LASTEXITCODE -ne 0) {
    Write-Error "LOSO training failed with exit code $LASTEXITCODE"
    exit 1
}

# 2. RUN SHIP TRAINING (Final model on all 450 clips)
Write-Host "[2/4] Running Ship training..."
vslr-train --data-dir $DataDir --recording-plan $Plan `
  --epochs 40 --augment 120 --batch-size 32 --learning-rate 0.001 `
  --seed 42 --num-workers 4 --model-dir $RunDir 2>&1 |
  Tee-Object "$RunDir\ship.log"

if ($LASTEXITCODE -ne 0) {
    Write-Error "Ship training failed with exit code $LASTEXITCODE"
    exit 1
}

# 3. VERIFY ARTIFACT PAIRING
Write-Host "[3/4] Verifying artifact pairing contract..."
$env:RUN_DIR = (Resolve-Path $RunDir).Path
@'
import json, os, sys
from pathlib import Path
from prototype_3_gestures.prepare_train import artifact_pair_matches
from prototype_3_gestures.vsl3.model import load_checkpoint

root = Path(os.environ["RUN_DIR"])
report = json.loads((root / "loso_report.json").read_text(encoding="utf-8"))
metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
checkpoint = root / "gesture_lstm.pt"

assert artifact_pair_matches(report["training_signature"], metrics, checkpoint)
model, labels, config = load_checkpoint(checkpoint)
assert len(labels) == 25
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

# 4. GENERATE CHARTS & EVALUATION REPORT
Write-Host "[4/4] Generating evaluation charts and markdown report..."
python scripts/generate_evaluation_artifacts.py "$RunDir"

Write-Host "========================================================="
Write-Host "ALL STEPS COMPLETED SUCCESSFULLY! Artifacts in: $RunDir"
Write-Host "========================================================="

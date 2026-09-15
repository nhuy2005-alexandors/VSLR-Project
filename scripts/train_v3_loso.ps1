param (
    [int]$Epochs = 13
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

Set-Location "D:\Dev\Workspaces\VSLR-Workspace"

$RunId = Get-Date -Format "yyyyMMdd-HHmmss"
$RunDir = "runs\v3-4signers-24-8clips-$RunId"
New-Item -ItemType Directory -Force $RunDir | Out-Null
Write-Host "========================================================="
Write-Host "STARTING V3 24-GESTURE 4-SIGNER 8-CLIPS LOSO RUN ($Epochs epochs): $RunDir"
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

# 0. RECORD ENVIRONMENT METADATA
git rev-parse HEAD | Set-Content "$RunDir\base-commit.txt"
git branch --show-current | Set-Content "$RunDir\branch.txt"
git status --porcelain | Set-Content "$RunDir\worktree-status.txt"
python -m pip freeze | Set-Content "$RunDir\pip-freeze.txt"
python --version | Set-Content "$RunDir\python-version.txt"
nvidia-smi | Set-Content "$RunDir\gpu-info.txt"

Assert-FilesExistAndNonEmpty $RunDir @(
    "base-commit.txt", "branch.txt", "worktree-status.txt", "pip-freeze.txt", "python-version.txt", "gpu-info.txt"
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

@"
LOSO Command:
vslr-train --data-dir dataset/recordings_v3_4x24_8 --recording-plan dataset/recording_plan_v3_4x24_8.json --labels-file dataset/labels_v2_24.txt --cache-dir dataset/processed/landmark_cache_v3_4x24_8 --loso --epochs $Epochs --augment 120 --batch-size 32 --learning-rate 0.001 --seed 42 --num-workers 4 --model-dir $RunDir
"@ | Set-Content "$RunDir\commands.txt" -Encoding utf8
Assert-FilesExistAndNonEmpty $RunDir @("commands.txt")

# 2. RUN LOSO TRAINING (4 Folds: P01, P02, P03, P04)
Write-Host "[2/5] Running LOSO 4-fold training on 768 clips for $Epochs epochs..."
vslr-train `
  --data-dir dataset/recordings_v3_4x24_8 `
  --recording-plan dataset/recording_plan_v3_4x24_8.json `
  --labels-file dataset/labels_v2_24.txt `
  --cache-dir dataset/processed/landmark_cache_v3_4x24_8 `
  --loso `
  --epochs $Epochs `
  --augment 120 `
  --batch-size 32 `
  --learning-rate 0.001 `
  --seed 42 `
  --num-workers 4 `
  --model-dir $RunDir 2>&1 | Tee-Object "$RunDir\loso.log"
Assert-StepSuccess "LOSO Training"
Assert-FilesExistAndNonEmpty $RunDir @("loso.log", "loso_report.json")

# 3. GENERATE EVALUATION ARTIFACTS
Write-Host "[3/5] Generating evaluation artifacts (curves, CM, per-label, report)..."
python scripts/generate_evaluation_artifacts.py $RunDir
Assert-StepSuccess "Generate Evaluation Artifacts"
Assert-FilesExistAndNonEmpty $RunDir @(
    "training_curves.png", "confusion_matrix.png", "per_label_accuracy.png", "evaluation_report.md"
)

# 4. GENERATE TOP-3 ANALYSIS
Write-Host "[4/5] Generating top-3 LOSO analysis..."
python scripts/generate_top3_loso_analysis.py $RunDir
Assert-StepSuccess "Generate Top-3 Analysis"
Assert-FilesExistAndNonEmpty (Join-Path $RunDir "top3_loso_analysis") @(
    "REPORT.md", "top3_predictions.csv", "top3_predictions.json", "top3_by_gesture.csv", "top3_by_gesture.json"
)

# 5. GENERATE CHECKSUMS AND RUN INDEPENDENT VERIFIER
Write-Host "[5/5] Generating recursive checksums (UTF-8 no BOM) and running independent verifier..."
python scripts/verify_v3_run.py $RunDir --generate-sums
Assert-StepSuccess "Independent Verifier & Checksums"

Write-Host "========================================================="
Write-Host "V3 LOSO RUN FULLY VERIFIED: $RunDir"
Write-Host "========================================================="

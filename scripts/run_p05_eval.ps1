param (
    [string]$OutputDir = "evaluation\p05-v3-final-20260916"
)

$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false

Set-Location "D:\Dev\Workspaces\VSLR-Workspace"

Write-Host "========================================================="
Write-Host "P05 FINAL EXTERNAL EVALUATION (ONE-WAY GATE)"
Write-Host "Target: $OutputDir"
Write-Host "========================================================="

$StartUtc = [DateTime]::UtcNow.ToString("o")
$GitHead = (git rev-parse HEAD).Trim()

$ModelPath = "runs\v3-ship-4signers-24-8clips-13ep-20260916-020145\gesture_lstm.pt"
$TrainManifestPath = "runs\v3-ship-4signers-24-8clips-13ep-20260916-020145\dataset_files_sha256.csv"
$RunManifestPath = "runs\v3-ship-4signers-24-8clips-13ep-20260916-020145\RUN_MANIFEST.json"

$ModelSha = (Get-FileHash -Algorithm SHA256 $ModelPath).Hash.ToLower()
$TrainManifestSha = (Get-FileHash -Algorithm SHA256 $TrainManifestPath).Hash.ToLower()
$RunManifestSha = (Get-FileHash -Algorithm SHA256 $RunManifestPath).Hash.ToLower()

New-Item -ItemType Directory -Force $OutputDir | Out-Null

$LogFile = "$OutputDir\execution.log"

@"
=========================================================
P05 FINAL EVALUATION EXECUTION LOG
Start UTC: $StartUtc
Git HEAD:  $GitHead
Model:     $ModelPath ($ModelSha)
TrainMan:  $TrainManifestPath ($TrainManifestSha)
RunMan:    $RunManifestPath ($RunManifestSha)
=========================================================
"@ | Set-Content $LogFile -Encoding utf8

$CommandStr = @"
vslr-eval --data-dir dataset/external_eval_v3 --model $ModelPath --training-manifest $TrainManifestPath --run-manifest $RunManifestPath --expected-signer P05 --clips-per-label 2 --allow-uncalibrated --confidence 0.50 --output-dir $OutputDir
"@
$CommandStr | Set-Content "$OutputDir\command.txt" -Encoding utf8

Write-Host "Executing frozen vslr-eval command..."
vslr-eval `
  --data-dir dataset/external_eval_v3 `
  --model $ModelPath `
  --training-manifest $TrainManifestPath `
  --run-manifest $RunManifestPath `
  --expected-signer P05 `
  --clips-per-label 2 `
  --allow-uncalibrated `
  --confidence 0.50 `
  --output-dir $OutputDir 2>&1 | Tee-Object -Append $LogFile

$ExitCode = $LASTEXITCODE
$EndUtc = [DateTime]::UtcNow.ToString("o")

@"
=========================================================
End UTC:   $EndUtc
Exit Code: $ExitCode
=========================================================
"@ | Add-Content $LogFile -Encoding utf8

Write-Host "vslr-eval finished with exit code $ExitCode"
if ($ExitCode -ne 0) {
    Write-Error "vslr-eval failed with exit code $ExitCode!"
    exit $ExitCode
}

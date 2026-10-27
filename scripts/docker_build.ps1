param([string]$Image = "credit-default-api")
$Root = Split-Path $PSScriptRoot -Parent

if (-not (Test-Path (Join-Path $Root "models/model.joblib"))) {
    Write-Host "models/model.joblib not found. Get it first: poetry run dvc pull (or poetry run dvc repro)."
    exit 1
}

docker build -t $Image $Root
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Built image: $Image"

# Webull Residual Liquidity Exhaustion Bot — one-shot Windows setup.
# Usage:  cd C:\webull_liquidity_bot ; Set-ExecutionPolicy -Scope Process Bypass ; .\setup.ps1
$ErrorActionPreference = "Stop"

Write-Host "== wliq setup ==" -ForegroundColor Cyan
py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[webull,dev]"

if (-not (Test-Path config\settings.yaml)) {
    Copy-Item config\settings.example.yaml config\settings.yaml
    Write-Host "Created config\settings.yaml from example." -ForegroundColor Yellow
}
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
New-Item -ItemType Directory -Force -Path logs, data | Out-Null

Write-Host "`nSet credentials for this session:" -ForegroundColor Yellow
Write-Host '  $env:WEBULL_APP_KEY="YOUR_APP_KEY"'
Write-Host '  $env:WEBULL_APP_SECRET="YOUR_APP_SECRET"'
Write-Host '  $env:WEBULL_ACCOUNT_ID="YOUR_ACCOUNT_ID"'

Write-Host "`nRunning diagnostics + tests..." -ForegroundColor Cyan
wliq doctor --config config\settings.yaml
pytest
Write-Host "`nSetup complete. Start recording with: wliq record --config config\settings.yaml" -ForegroundColor Green

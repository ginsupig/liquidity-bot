# Start the bot in the mode set in config\settings.yaml (default: record).
$ErrorActionPreference = "Stop"
& .\.venv\Scripts\Activate.ps1
if (-not $env:WEBULL_APP_KEY) { Write-Host "WEBULL_APP_KEY not set" -ForegroundColor Red; exit 1 }
wliq run --config config\settings.yaml

# Runs the live ingest loop from the project folder with its virtual environment.
# The map key is read from FIRMS_MAP_KEY (environment or the .env file in the project folder).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
& (Join-Path $root ".venv\Scripts\Activate.ps1")
python -m src.ingest --loop --every-minutes 180

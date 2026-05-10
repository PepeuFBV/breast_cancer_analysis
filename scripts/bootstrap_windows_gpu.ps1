param(
    [string]$VenvDir = ".venv",
    [switch]$InstallDev
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Description,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Script
    )
    Write-Host $Description
    & $Script
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

Invoke-Step -Description "Checking Python 3.10 launcher..." -Script {
    & py -3.10 -V | Out-Host
}

Invoke-Step -Description "Creating virtual environment at $VenvDir" -Script {
    & py -3.10 -m venv $VenvDir
}

$venvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment python not found at $venvPython"
}

Invoke-Step -Description "Upgrading pip/setuptools/wheel..." -Script {
    & $venvPython -m pip install --upgrade pip setuptools wheel
}

Invoke-Step -Description "Installing Windows native GPU requirements..." -Script {
    & $venvPython -m pip install -r "requirements-windows-gpu.txt"
}

if ($InstallDev) {
    Invoke-Step -Description "Installing development requirements..." -Script {
        & $venvPython -m pip install -r "requirements-dev.txt"
    }
}

Invoke-Step -Description "Installing project in editable mode..." -Script {
    & $venvPython -m pip install -e .
}

Invoke-Step -Description "Running native Windows GPU checker..." -Script {
    & $venvPython "scripts/check_windows_gpu.py"
}

Write-Host "Done. Activate with:"
Write-Host "  $VenvDir\Scripts\Activate.ps1"

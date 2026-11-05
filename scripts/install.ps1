<#
.SYNOPSIS
  LeakLens quick-start — safe Windows PowerShell installer (no admin, no destructive actions).

.DESCRIPTION
  Checks Python, creates or reuses a virtual environment, installs LeakLens,
  verifies the installation, and prints usage instructions.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1
  powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Extras "arena,parquet"
#>
[CmdletBinding()]
param(
    [string]$VenvDir = ".venv",
    [string]$Extras = ""
)

$ErrorActionPreference = "Stop"

Write-Host "LeakLens installer (Windows)"
Write-Host "==========================="

# 1. Check Python
$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) {
    $pyVer = & py -3 --version 2>&1
    Write-Host "Found: $pyVer ($($py.Source))"
} else {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        throw "Python >=3.10 was not found. Install it from https://www.python.org/downloads/"
    }
    $pyVer = & $python.Source --version 2>&1
    Write-Host "Found: $pyVer ($($python.Source))"
}
if ($pyVer -notmatch "Python (\d+)\.(\d+)") { throw "Could not parse Python version: $pyVer" }
$major, $minor = [int]$Matches[1], [int]$Matches[2]
if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
    throw "Python >=3.10 is required (found $pyVer). Install from https://www.python.org/downloads/"
}
Write-Host "Python OK ($pyVer)"

# 2. Create or reuse venv (nothing is deleted)
if (-not (Test-Path -LiteralPath "$VenvDir\Scripts\python.exe")) {
    Write-Host "Creating virtual environment at $VenvDir ..."
    if ($py) { & py -3 -m venv $VenvDir } else { python -m venv $VenvDir }
}
$pyExe = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pyExe)) { throw "Virtual environment python not found at $pyExe" }
Write-Host "Using: $pyExe ($(& $pyExe --version))"

# 3. Install LeakLens (core; extras only if requested)
$target = "leaklens-ml"
if ($Extras) { $target = "leaklens-ml[$Extras]" }
Write-Host "Installing $target ..."
& $pyExe -m pip install --upgrade $target
if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)." }

# 4. Verify installation
$ver = & $pyExe -m leaklens --version
if ($LASTEXITCODE -ne 0) { throw "leaklens --version failed (exit $LASTEXITCODE)." }
Write-Host "leaklens --version => $ver"
& $pyExe -m leaklens doctor
if ($LASTEXITCODE -ne 0) { throw "leaklens doctor failed (exit $LASTEXITCODE); see above." }

Write-Host ""
Write-Host "Done. Usage examples:"
Write-Host "  $pyExe -m leaklens --scan ."
Write-Host "  $pyExe -m leaklens demo.csv --target is_fraud --arena   # needs [arena]"
Write-Host "  $pyExe -m leaklens --help"
Write-Host "  $pyExe -m leaklens doctor"
Write-Host ""
Write-Host "Activate with:  .\$VenvDir\Scripts\Activate.ps1"

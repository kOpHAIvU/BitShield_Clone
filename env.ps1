# PowerShell environment setup script for BitShield project
# Source this script to set up the environment

# Get the script directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

# Set project paths
$env:TVM_DIR = Join-Path $SCRIPT_DIR "compilers\tvm-main"
$env:NNFUSION_DIR = Join-Path $SCRIPT_DIR "compilers\nnfusion-main"
$env:TOOLS_DIR = Join-Path $SCRIPT_DIR "tools"

# Set virtual environment path
$env:VENV_DIR = Join-Path $SCRIPT_DIR "venv"

# Change to project directory
Set-Location $SCRIPT_DIR

# Check if virtual environment exists
if (-not (Test-Path $env:VENV_DIR)) {
    Write-Host "Virtual environment not found. Please run setup.ps1 first." -ForegroundColor Red
    exit 1
}

# Activate virtual environment
& (Join-Path $env:VENV_DIR "Scripts\Activate.ps1")
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to activate virtual environment." -ForegroundColor Red
    exit 1
}

# Add tools directory to PATH
$env:PATH = "$env:TOOLS_DIR;$env:PATH"

# Set Python path for TVM and NNFusion
$env:PYTHONPATH = "$env:TVM_DIR\python;$env:NNFUSION_DIR\src\python;$env:PYTHONPATH"

# Set torch home
$env:TORCH_HOME = Join-Path $SCRIPT_DIR ".cache\torch"

# Set ImageNet root if not already set
if (-not $env:IMAGENET_ROOT) {
    Write-Host "Warning: IMAGENET_ROOT not set. Some functionality may not work." -ForegroundColor Yellow
    Write-Host "Please set it to your ImageNet dataset path." -ForegroundColor Yellow
}

Write-Host "Environment activated successfully!" -ForegroundColor Green
Write-Host "Python: $env:PYTHONPATH" -ForegroundColor Cyan
Write-Host "Tools: $env:TOOLS_DIR" -ForegroundColor Cyan
Write-Host "" 
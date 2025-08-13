# PowerShell setup script for BitShield project
# Run this script in PowerShell with: .\setup.ps1

param(
    [switch]$Force
)

Write-Host "Setting up BitShield project for Windows..." -ForegroundColor Green

# Check if Python 3.8+ is available
try {
    $pythonVersion = python --version 2>&1
    if ($pythonVersion -match "Python 3\.(8|9|10|11|12)") {
        Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "Python 3.8+ is required. Found: $pythonVersion" -ForegroundColor Red
        Write-Host "You can download it from https://www.python.org/downloads/" -ForegroundColor Yellow
        exit 1
    }
} catch {
    Write-Host "Python not found. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

# Initialize git submodules
Write-Host "Initializing git submodules..." -ForegroundColor Yellow
git submodule update --init --recursive
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to initialize git submodules." -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv") -or $Force) {
    if ($Force -and (Test-Path "venv")) {
        Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force "venv"
    }
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment and install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to activate virtual environment." -ForegroundColor Red
    exit 1
}

python -m pip install --upgrade pip
python -m pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements_windows.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install dependencies." -ForegroundColor Red
    exit 1
}

# Create necessary directories
Write-Host "Creating necessary directories..." -ForegroundColor Yellow
$directories = @(
    "datasets",
    "models", 
    "built",
    "results",
    ".cache",
    "built-aux",
    "ghidra\db",
    "ghidra\analysis"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir" -ForegroundColor Green
    }
}

# Check for ImageNet environment variable
if (-not $env:IMAGENET_ROOT) {
    Write-Host "Warning: IMAGENET_ROOT environment variable is not set." -ForegroundColor Yellow
    Write-Host "Please set it to the path of your ImageNet dataset." -ForegroundColor Yellow
    Write-Host "Example: `$env:IMAGENET_ROOT = 'C:\path\to\imagenet'" -ForegroundColor Cyan
}

Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Set IMAGENET_ROOT environment variable if you have ImageNet dataset" -ForegroundColor White
Write-Host "2. Run: .\env.ps1" -ForegroundColor White
Write-Host "3. Run: python tools\ensure_datasets.py" -ForegroundColor White
Write-Host "4. For building models, you'll need Docker Desktop installed" -ForegroundColor White
Write-Host "" 
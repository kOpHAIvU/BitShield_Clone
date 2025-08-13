# Windows Compatibility Changes

This document summarizes the changes made to make the BitShield project compatible with Windows.

## Files Added

### Setup and Environment Scripts
- `setup.bat` - Windows batch script for initial setup
- `setup.ps1` - PowerShell script for initial setup (alternative)
- `env.bat` - Windows batch script for environment activation
- `env.ps1` - PowerShell script for environment activation (alternative)
- `quick_start_windows.bat` - Quick start guide for Windows users

### Docker Scripts for Windows
- `docker/setup.bat` - Windows version of docker/setup.sh
- `docker/run-in-docker.bat` - Windows version of docker/run-in-docker.sh
- `docker/config.bat` - Windows version of docker/config.sh

### Tool Scripts for Windows
- `tools/runattacksim.bat` - Windows version of tools/runattacksim.sh

### Configuration Files
- `requirements_windows.txt` - Windows-compatible requirements
- `dvc_windows.yaml` - Windows-compatible DVC configuration
- `README_Windows.md` - Comprehensive Windows setup guide

## Key Changes Made

### 1. Script Conversion
- Converted all bash scripts (`.sh`) to Windows batch scripts (`.bat`)
- Created PowerShell alternatives (`.ps1`) for better Windows integration
- Updated path separators from `/` to `\` for Windows
- Fixed environment variable syntax for Windows

### 2. Python Environment
- Created `requirements_windows.txt` with Windows-compatible package versions
- Added `pywin32` dependency for Windows-specific functionality
- Updated PyTorch installation to use CPU-only version (more compatible)

### 3. Docker Integration
- Updated Docker commands to work with Windows paths
- Fixed volume mounting syntax for Windows
- Added Windows-specific error handling

### 4. File Paths
- Updated all hardcoded Unix paths to use Windows path separators
- Made path handling cross-platform compatible where possible
- Updated DVC configuration to use Windows path syntax

### 5. Environment Variables
- Updated environment variable syntax for Windows
- Added proper Windows PATH handling
- Fixed Python path configuration for Windows

## Usage Instructions

### For Command Prompt Users
```cmd
# Initial setup
setup.bat

# Activate environment
env.bat

# Quick start (recommended for new users)
quick_start_windows.bat
```

### For PowerShell Users
```powershell
# Initial setup
.\setup.ps1

# Activate environment
.\env.ps1
```

### Docker Setup (Required for Model Building)
```cmd
# Build Docker image
docker\setup.bat

# Run commands in Docker
docker\run-in-docker.bat python tools\ensure_datasets.py
```

## Prerequisites

1. **Python 3.8** - Must be installed and in PATH
2. **Git** - For submodule management
3. **Docker Desktop** - For model building and experiments
4. **ImageNet Dataset** (Optional) - Set `IMAGENET_ROOT` environment variable

## Troubleshooting

### Common Issues
1. **Python not found** - Ensure Python 3.8 is installed and in PATH
2. **Docker not running** - Start Docker Desktop before running Docker commands
3. **Permission errors** - Run Command Prompt as Administrator
4. **Path issues** - Ensure all paths use Windows separators (`\`)

### Performance Tips
1. Use SSD storage for better I/O performance
2. Allocate 8GB+ RAM to Docker Desktop
3. Use WSL2 backend for Docker (better performance)

## File Structure

```
BitShield_Clone/
├── setup.bat                    # Windows setup script
├── setup.ps1                    # PowerShell setup script
├── env.bat                      # Windows environment script
├── env.ps1                      # PowerShell environment script
├── quick_start_windows.bat      # Quick start guide
├── requirements_windows.txt     # Windows-compatible requirements
├── dvc_windows.yaml            # Windows DVC configuration
├── README_Windows.md           # Windows setup guide
├── WINDOWS_CHANGES.md          # This file
├── docker/
│   ├── setup.bat               # Docker setup for Windows
│   ├── run-in-docker.bat       # Docker runner for Windows
│   └── config.bat              # Docker config for Windows
└── tools/
    └── runattacksim.bat        # Attack simulation for Windows
```

## Notes

- All original Linux/Unix scripts remain unchanged
- Windows scripts are additions, not replacements
- The project can still be run on Linux/Unix using the original scripts
- Docker is still required for full functionality (model building, experiments)
- Some features may require WSL2 for optimal performance on Windows

## Testing

The Windows compatibility has been tested for:
- ✅ Environment setup and activation
- ✅ Python package installation
- ✅ Dataset downloading
- ✅ Basic Python script execution
- ✅ Docker integration (requires Docker Desktop)
- ✅ DVC operations

## Support

For Windows-specific issues:
1. Check `README_Windows.md` for detailed instructions
2. Verify all prerequisites are installed correctly
3. Ensure Docker Desktop is running
4. Check file permissions and run as Administrator if needed

For general project issues, refer to the main `README.md` file. 
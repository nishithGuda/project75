# PowerShell script to process and integrate Rico datasets for UI Navigation Assistant project

# Parse command line parameters
param (
    [string]$ricoBase,
    [string]$ricoSemantics,
    [switch]$useRefexp,
    [string]$outputDir = "C:/Users/nstar/Desktop/project/backend/data/processed_data"
)

# Check for Python
try {
    python --version
} catch {
    Write-Host "Error: Python is required but not found." -ForegroundColor Red
    exit 1
}

# Create required directories
New-Item -ItemType Directory -Force -Path utils | Out-Null
New-Item -ItemType Directory -Force -Path data\training | Out-Null

# Check if data processor files exist
$filesExist = $true
$requiredFiles = @(
    "data_processor.py",
    "utils\metadata_parser.py",
    "utils\semantic_parser.py",
    "utils\refexp_parser.py",
    "utils\dataset_integrator.py"
)

foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        $filesExist = $false
        Write-Host "Missing file: $file" -ForegroundColor Yellow
    }
}

if (-not $filesExist) {
    Write-Host "Error: Required processor files not found." -ForegroundColor Red
    Write-Host "Please ensure the following files exist:"
    foreach ($file in $requiredFiles) {
        Write-Host "  - $file"
    }
    exit 1
}

# Check for required arguments
if ([string]::IsNullOrEmpty($ricoBase) -and [string]::IsNullOrEmpty($ricoSemantics) -and -not $useRefexp) {
    Write-Host "Warning: No datasets specified. At least one dataset is required." -ForegroundColor Yellow
    Write-Host "Usage: .\process_datasets.ps1 -ricoBase <path> -ricoSemantics <path> -useRefexp -outputDir <path>"
    exit 1
}

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Green
if (Test-Path "..\requirements.txt") {
    pip install -r ..\requirements.txt
} elseif (Test-Path "requirements.txt") {
    pip install -r requirements.txt
} else {
    Write-Host "Warning: requirements.txt not found. Installing essential packages..." -ForegroundColor Yellow
    pip install numpy tqdm datasets
}

# Build command
$cmd = "python data_processor.py"

if (-not [string]::IsNullOrEmpty($ricoBase)) {
    $cmd += " --rico-base `"$ricoBase`""
}

if (-not [string]::IsNullOrEmpty($ricoSemantics)) {
    $cmd += " --rico-semantics `"$ricoSemantics`""
}

if ($useRefexp) {
    $cmd += " --use-refexp"
}

$cmd += " --output-dir `"$outputDir`" --copy-to-project"

# Run processor
Write-Host "Processing datasets with command:" -ForegroundColor Green
Write-Host $cmd
Write-Host ""

Invoke-Expression $cmd

# Check if processing was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nDataset processing completed successfully!" -ForegroundColor Green
    Write-Host "Processed data has been saved to $outputDir"
    Write-Host "Project files have been copied to the data directory"
} else {
    Write-Host "`nDataset processing failed." -ForegroundColor Red
    Write-Host "Please check the error messages above."
    exit 1
}
# This script zips your HistoMoE training code so it can be uploaded securely to Kaggle as a private Dataset.
# It explicitly ignores your virtual environment, Git history, outputs, and cache files to keep the zip very small and completely private.

$compressArgs = @{
    Path = (
        ".\histomoe", 
        ".\train.py", 
        ".\evaluate.py", 
        ".\api.py",
        ".\app.py",
        ".\requirements.txt", 
        ".\requirements-dev.txt",
        ".\pyproject.toml",
        ".\configs"
    )
    DestinationPath = ".\histomoe_code.zip"
    Force = $true
}

Write-Host "Zipping HistoMoE codebase for Kaggle (ignoring .venv and .git)..." -ForegroundColor Cyan
Compress-Archive @compressArgs
Write-Host "Done! You can now upload 'histomoe_code.zip' to a Kaggle Notebook." -ForegroundColor Green
Write-Host "No private local settings or heavy environments were included." -ForegroundColor Yellow

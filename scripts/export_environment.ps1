# Run this while the known-good project virtual environment is activated.
# It records package versions without committing the virtual environment itself.
python -m pip freeze | Out-File -Encoding utf8 requirements-lock-windows.txt
Write-Host "Created requirements-lock-windows.txt"

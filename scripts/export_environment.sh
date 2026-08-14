#!/usr/bin/env sh
# Run this while the known-good project virtual environment is activated.
python -m pip freeze > requirements-lock-linux.txt
echo "Created requirements-lock-linux.txt"

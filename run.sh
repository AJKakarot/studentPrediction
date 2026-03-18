#!/bin/bash
# Run with the Python that has packages installed (from pip install)
cd "$(dirname "$0")"

# Try Python 3.14 first (where pip installed packages)
if command -v python3.14 &>/dev/null; then
    exec python3.14 run_all.py "$@"
fi
# Fallback: use same python as pip
if command -v pip3 &>/dev/null; then
    exec "$(pip3 show pandas 2>/dev/null | grep 'Location' | cut -d' ' -f2 | xargs dirname)/../../bin/python3" run_all.py "$@" 2>/dev/null || true
fi
# Last resort
exec python3 run_all.py "$@"

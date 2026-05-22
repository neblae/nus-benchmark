#!/usr/bin/env bash
# Helper to initialize the repo and push to GitHub.
# 1) Create an empty repo on GitHub (no README/license — we have those).
# 2) Run this from the project root, passing your repo URL:
#       bash scripts/init_git.sh git@github.com:USERNAME/nus-benchmark.git
set -euo pipefail
REMOTE="${1:?Pass your GitHub repo URL as the first argument}"
git init
git add .
git commit -m "Initial commit: NUS schedule benchmarking pipeline for HSQC"
git branch -M main
git remote add origin "$REMOTE"
git push -u origin main
echo "Pushed to $REMOTE"

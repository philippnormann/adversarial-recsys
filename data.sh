#!/usr/bin/env bash
set -e

python -m pipenv run python src/deepfashion/download.py
src/deepfashion/resize.sh
python -m pipenv run python src/deepfashion/prepare.py
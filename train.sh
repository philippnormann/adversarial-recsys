#!/usr/bin/env bash
set -e

pipenv run python -m src.train --batch-size 128 adversarial --num-epochs 12

#!/usr/bin/env bash
set -e

# normal-24-epochs
pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.01 --no-plot cw
pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.02 --no-plot cw
pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.03 --no-plot cw
pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.04 --no-plot cw
pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.05 --no-plot cw

pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.01 --no-plot pgd
pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.02 --no-plot pgd
pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.03 --no-plot pgd
pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.04 --no-plot pgd
pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.05 --no-plot pgd

pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.01 --no-plot fgsm
pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.02 --no-plot fgsm
pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.03 --no-plot fgsm
pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.04 --no-plot fgsm
pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.05 --no-plot fgsm

pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.01 --no-plot cw
pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.02 --no-plot cw
pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.03 --no-plot cw
pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.04 --no-plot cw
pipenv run python -m src.attack --model-name normal-24-epochs --epsilon 0.05 --no-plot cw

# adversarial-24-epochs
pipenv run python -m src.attack --model-name adversarial-24-epochs --epsilon 0.01 --no-plot pgd
pipenv run python -m src.attack --model-name adversarial-24-epochs --epsilon 0.05 --no-plot pgd
pipenv run python -m src.attack --model-name adversarial-24-epochs --epsilon 0.1 --no-plot pgd
pipenv run python -m src.attack --model-name adversarial-24-epochs --epsilon 0.2 --no-plot pgd
pipenv run python -m src.attack --model-name adversarial-24-epochs --epsilon 0.3 --no-plot pgd

pipenv run python -m src.attack --model-name adversarial-24-epochs --epsilon 0.01 --no-plot fgsm
pipenv run python -m src.attack --model-name adversarial-24-epochs --epsilon 0.05 --no-plot fgsm
pipenv run python -m src.attack --model-name adversarial-24-epochs --epsilon 0.1 --no-plot fgsm
pipenv run python -m src.attack --model-name adversarial-24-epochs --epsilon 0.2 --no-plot fgsm
pipenv run python -m src.attack --model-name adversarial-24-epochs --epsilon 0.3 --no-plot fgsm

pipenv run python -m src.attack --model-name adversarial-24-epochs --epsilon 0.01 --no-plot cw
pipenv run python -m src.attack --model-name adversarial-24-epochs --epsilon 0.05 --no-plot cw
pipenv run python -m src.attack --model-name adversarial-24-epochs --epsilon 0.1 --no-plot cw
pipenv run python -m src.attack --model-name adversarial-24-epochs --epsilon 0.2 --no-plot cw
pipenv run python -m src.attack --model-name adversarial-24-epochs --epsilon 0.3 --no-plot cw

# curriculum-adversarial-8-k
pipenv run python -m src.attack --model-name curriculum-adversarial-8-k --epsilon 0.01 --no-plot pgd
pipenv run python -m src.attack --model-name curriculum-adversarial-8-k --epsilon 0.05 --no-plot pgd
pipenv run python -m src.attack --model-name curriculum-adversarial-8-k --epsilon 0.1 --no-plot pgd
pipenv run python -m src.attack --model-name curriculum-adversarial-8-k --epsilon 0.2 --no-plot pgd
pipenv run python -m src.attack --model-name curriculum-adversarial-8-k --epsilon 0.3 --no-plot pgd

pipenv run python -m src.attack --model-name curriculum-adversarial-8-k --epsilon 0.01 --no-plot fgsm
pipenv run python -m src.attack --model-name curriculum-adversarial-8-k --epsilon 0.05 --no-plot fgsm
pipenv run python -m src.attack --model-name curriculum-adversarial-8-k --epsilon 0.1 --no-plot fgsm
pipenv run python -m src.attack --model-name curriculum-adversarial-8-k --epsilon 0.2 --no-plot fgsm
pipenv run python -m src.attack --model-name curriculum-adversarial-8-k --epsilon 0.3 --no-plot fgsm

pipenv run python -m src.attack --model-name curriculum-adversarial-8-k --epsilon 0.01 --no-plot cw
pipenv run python -m src.attack --model-name curriculum-adversarial-8-k --epsilon 0.05 --no-plot cw
pipenv run python -m src.attack --model-name curriculum-adversarial-8-k --epsilon 0.1 --no-plot cw
pipenv run python -m src.attack --model-name curriculum-adversarial-8-k --epsilon 0.2 --no-plot cw
pipenv run python -m src.attack --model-name curriculum-adversarial-8-k --epsilon 0.3 --no-plot cw

#!/usr/bin/env bash
set -e

EVAL_SCRIPT="pipenv run python -m src.evaluate"

function evaluateKnnAttack() {
    local modelName="$1"
    local batchSize="$2"
    local numSamples="$3"
    local iterations="$4"
    local attack="$5"
    shift 5
    local epsilons=("$@")

    $EVAL_SCRIPT \
        --model-name "$modelName" \
        --batch-size "$batchSize" \
        --split test knn --num-samples "$numSamples" \
        --num-iterations "$iterations" \
        --attack "$attack" \
        --epsilons "${epsilons[@]}"
}

function evaluateKnn() {
    local modelName="$1"
    local batchSize="$2"
    shift 2
    local epsilons=("$@")

    evaluateKnnAttack "$modelName" "$batchSize" 10000 1 fgsm "${epsilons[@]}"
    evaluateKnnAttack "$modelName" "$batchSize" 10000 8 pgd "${epsilons[@]}"
    evaluateKnnAttack "$modelName" "$batchSize" 10000 16 pgd "${epsilons[@]}"
    evaluateKnnAttack "$modelName" "$batchSize" 10000 32 pgd "${epsilons[@]}"
    evaluateKnnAttack "$modelName" "$batchSize" 10000 64 pgd "${epsilons[@]}"
    evaluateKnnAttack "$modelName" "$batchSize" 10000 128 pgd "${epsilons[@]}"
    evaluateKnnAttack "$modelName" "$batchSize" 1000 1000 cw "${epsilons[@]}"
}

evaluateKnn normal-24-epochs 64 0.001 0.002 0.003 0.004 0.005
evaluateKnn adversarial-24-epochs 64 0.01 0.05 0.1 0.2 0.3
evaluateKnn curriculum-adversarial-8-k 64 0.01 0.05 0.1 0.2 0.3

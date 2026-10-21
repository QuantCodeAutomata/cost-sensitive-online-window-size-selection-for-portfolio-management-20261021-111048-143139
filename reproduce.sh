#!/usr/bin/env bash
# reproduce.sh — Run all experiments for
# "Cost-Sensitive Online Window Size Selection for Portfolio Management"
#
# Usage:
#   ./reproduce.sh            # run all experiments
#   ./reproduce.sh --exp1     # run only exp_1 (synthetic, no API needed)
#   ./reproduce.sh --exp4     # run only exp_4 (additional synthetic, no API needed)
#   ./reproduce.sh --tests    # run only unit tests
#
# Required environment variables for exp_2 and exp_3 (historical data):
#   DATALOADER_API_KEY   — from https://quantx.limex.pro/
#   MASSIVE_API_KEY      — from https://massive.com/
#
# Experiments that do NOT require API keys: exp_1, exp_4 (synthetic data only)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Cost-Sensitive Online Window Size Selection — Reproduction"
echo "============================================================"
echo ""

# Parse arguments
RUN_ALL=true
RUN_EXP1=false
RUN_EXP2=false
RUN_EXP3=false
RUN_EXP4=false
RUN_TESTS=false

for arg in "$@"; do
    case $arg in
        --exp1) RUN_EXP1=true; RUN_ALL=false ;;
        --exp2) RUN_EXP2=true; RUN_ALL=false ;;
        --exp3) RUN_EXP3=true; RUN_ALL=false ;;
        --exp4) RUN_EXP4=true; RUN_ALL=false ;;
        --tests) RUN_TESTS=true; RUN_ALL=false ;;
        --all) RUN_ALL=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

if [ "$RUN_ALL" = true ]; then
    RUN_EXP1=true
    RUN_EXP2=true
    RUN_EXP3=true
    RUN_EXP4=true
    RUN_TESTS=true
fi

# Create results directory
mkdir -p results

# ── Unit Tests ──────────────────────────────────────────────────────────────
if [ "$RUN_TESTS" = true ]; then
    echo "--- Running unit tests ---"
    python -m pytest tests/ -v --tb=short 2>&1 | tee results/test_output.txt
    TEST_STATUS=${PIPESTATUS[0]}
    if [ $TEST_STATUS -ne 0 ]; then
        echo "WARNING: Some tests failed. Check results/test_output.txt"
    else
        echo "All tests passed."
    fi
    echo ""
fi

# ── Exp 1: Synthetic Regime-Shift ───────────────────────────────────────────
if [ "$RUN_EXP1" = true ]; then
    echo "--- Running Exp 1: Synthetic Regime-Shift ---"
    python exp/exp1_synthetic.py 2>&1 | tee results/exp1_log.txt
    echo "Exp 1 done. Artifacts in results/exp1_*"
    echo ""
fi

# ── Exp 4: Additional Synthetic Analyses ────────────────────────────────────
if [ "$RUN_EXP4" = true ]; then
    echo "--- Running Exp 4: Additional Synthetic Analyses ---"
    python exp/exp4_sensitivity.py 2>&1 | tee results/exp4_log.txt
    echo "Exp 4 done. Artifacts in results/exp4_*"
    echo ""
fi

# ── Exp 2: DJIA Historical Study ────────────────────────────────────────────
if [ "$RUN_EXP2" = true ]; then
    if [ -z "${DATALOADER_API_KEY:-}" ] || [ -z "${MASSIVE_API_KEY:-}" ]; then
        echo "WARNING: DATALOADER_API_KEY and/or MASSIVE_API_KEY not set."
        echo "Skipping Exp 2 (DJIA historical study requires market data API)."
    else
        echo "--- Running Exp 2: DJIA Historical Study ---"
        python exp/exp2_djia.py 2>&1 | tee results/exp2_log.txt
        echo "Exp 2 done. Artifacts in results/exp2_*"
    fi
    echo ""
fi

# ── Exp 3: S&P 500 Historical Study ─────────────────────────────────────────
if [ "$RUN_EXP3" = true ]; then
    if [ -z "${DATALOADER_API_KEY:-}" ] || [ -z "${MASSIVE_API_KEY:-}" ]; then
        echo "WARNING: DATALOADER_API_KEY and/or MASSIVE_API_KEY not set."
        echo "Skipping Exp 3 (S&P 500 historical study requires market data API)."
    else
        echo "--- Running Exp 3: S&P 500 Historical Study ---"
        python exp/exp3_sp500.py 2>&1 | tee results/exp3_log.txt
        echo "Exp 3 done. Artifacts in results/exp3_*"
    fi
    echo ""
fi

echo "============================================================"
echo "Reproduction complete. See results/RESULTS.md for summary."
echo "============================================================"

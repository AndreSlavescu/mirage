#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/paper_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "       PDL Paper Benchmark Suite"
echo "=============================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Timestamp: ${TIMESTAMP}"
echo ""

cd "${SCRIPT_DIR}"

echo "[1/2] Running characterization study..."
echo "----------------------------------------------"
python -u characterization.py --output "${OUTPUT_DIR}/characterization_${TIMESTAMP}.json" 2>&1 | tee "${OUTPUT_DIR}/characterization_${TIMESTAMP}.log"

echo ""
echo "[2/2] Running full experiment suite..."
echo "----------------------------------------------"
python -u run_experiments.py --output "${OUTPUT_DIR}" 2>&1 | tee "${OUTPUT_DIR}/experiments_${TIMESTAMP}.log"

echo ""
echo "=============================================="
echo "           Benchmark Complete"
echo "=============================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
ls -1 "${OUTPUT_DIR}"/*.json "${OUTPUT_DIR}"/*.tex 2>/dev/null || echo "  (no result files yet)"


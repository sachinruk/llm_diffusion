#!/usr/bin/env bash

set -euo pipefail

# Script to run CLIP-JEPA training and shutdown the machine when complete
# This script reads hyperparameters from config.yaml and runs training

echo "========================================"
echo "CLIP-JEPA Training Script"
echo "========================================"
echo "Start time: $(date)"
echo ""

# Configuration
SCRIPT_DIR="$(git rev-parse --show-toplevel)"
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"
LOG_FILE="${SCRIPT_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

echo "Script directory: ${SCRIPT_DIR}"
echo "Config file: ${CONFIG_FILE}"
echo "Log file: ${LOG_FILE}"
echo ""

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

echo "Converting YAML config to JSON..."
# Convert YAML to JSON using Python (yq might not be available)
HYPERPARAMETERS_JSON=$(python3 << EOF
import yaml
import json
import sys

try:
    with open('${CONFIG_FILE}', 'r') as f:
        config = yaml.safe_load(f)
    print(json.dumps(config))
except Exception as e:
    print(f"Error converting YAML to JSON: {e}", file=sys.stderr)
    sys.exit(1)
EOF
)

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to convert YAML to JSON"
    exit 1
fi

echo "Configuration loaded successfully"
echo ""
echo "========================================"
echo "Starting Training"
echo "========================================"
echo ""

# Run training and capture output to both console and log file
cd "${SCRIPT_DIR}"
source .venv/bin/activate
python3 -m src.main --hyper-parameters-json "${HYPERPARAMETERS_JSON}" 2>&1 | tee "${LOG_FILE}"

# Check if training succeeded
TRAINING_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================"
echo "Training Complete"
echo "========================================"
echo "End time: $(date)"
echo "Exit code: ${TRAINING_EXIT_CODE}"
echo "Log saved to: ${LOG_FILE}"
echo ""

if [ ${TRAINING_EXIT_CODE} -ne 0 ]; then
    echo "WARNING: Training exited with non-zero status code: ${TRAINING_EXIT_CODE}"
    echo "Check the log file for details: ${LOG_FILE}"
    echo ""
    echo "Shutdown CANCELLED due to training failure"
    exit ${TRAINING_EXIT_CODE}
fi

echo "Training completed successfully!"
echo ""
echo "========================================"
echo "Initiating System Shutdown"
echo "========================================"
echo "Shutting down in 60 seconds..."
echo "Press Ctrl+C to cancel shutdown"
echo ""

# Give user time to cancel if needed
sleep 10
echo "Shutting down in 50 seconds..."
sleep 10
echo "Shutting down in 40 seconds..."
sleep 10
echo "Shutting down in 30 seconds..."
sleep 10
echo "Shutting down in 20 seconds..."
sleep 10
echo "Shutting down in 10 seconds..."
sleep 10

echo "Shutting down NOW..."
sudo shutdown -h now


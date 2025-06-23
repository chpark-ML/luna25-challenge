#!/bin/bash

# Image Level Model Analysis Script
# This script runs comprehensive failure analysis and advanced ensemble analysis

set -e

echo "=========================================="
echo "Image Level Model Analysis"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "main_inference.py" ]; then
    echo "Error: Please run this script from the analyzer/image_level directory"
    exit 1
fi

# Create output directories
mkdir -p failure_analysis_output
mkdir -p advanced_ensemble_output

echo "🔍 Running Failure Analysis..."
python failure_analysis.py

echo ""
echo "🚀 Running Advanced Ensemble Analysis..."
python advanced_ensemble.py

echo ""
echo "✅ Analysis Complete!"
echo ""
echo "📁 Output directories:"
echo "   - failure_analysis_output/"
echo "   - advanced_ensemble_output/"
echo ""
echo "📄 Reports generated:"
echo "   - failure_analysis_output/failure_analysis_report.md"
echo "   - advanced_ensemble_output/advanced_ensemble_report.md"
echo ""
echo "📊 Visualizations generated in both output directories" 
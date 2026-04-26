#!/usr/bin/env bash
# NeuralLens one-command Mac setup
set -e

echo "🧠 NeuralLens Setup"
echo "==================="

# Check Python 3.10+
python3 --version

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Copy env template if .env doesn't exist
if [ ! -f .env ]; then
  cp .env.example .env
  echo "⚠️  Created .env from template — fill in your API keys"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Fill in .env with your API keys"
echo "  2. Run: source .venv/bin/activate"
echo "  3. Run: bash run_all.sh"

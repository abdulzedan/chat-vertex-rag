#!/bin/bash

# Setup Development Environment for RAG Demo
# This script sets up pre-commit hooks and development dependencies

set -e  # Exit on any error

echo "🚀 Setting up RAG Demo development environment..."

# Check if we're in the right directory
if [ ! -f ".pre-commit-config.yaml" ]; then
    echo "❌ Error: .pre-commit-config.yaml not found. Please run this script from the project root."
    exit 1
fi

# Backend setup
echo "📦 Setting up backend dependencies..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

cd ..

# Frontend setup
echo "📦 Setting up frontend dependencies..."
cd frontend

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
npm install

cd ..

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
source backend/venv/bin/activate
pre-commit install
pre-commit install --hook-type commit-msg

# Run initial pre-commit check
echo "🧪 Running initial code quality checks..."
pre-commit run --all-files || {
    echo "⚠️  Some pre-commit checks failed. This is normal for initial setup."
    echo "   Re-run 'pre-commit run --all-files' to see if issues were auto-fixed."
}

echo "✅ Development environment setup complete!"
echo ""
echo "📝 Quick start commands:"
echo "  Backend:  cd backend && source venv/bin/activate && uvicorn app.main:app --reload"
echo "  Frontend: cd frontend && npm run dev"
echo "  Linting:  pre-commit run --all-files"
echo ""
echo "🎯 Git hooks are now active! Code will be automatically checked on commit."
#!/bin/bash
# =============================================================================
# Resonantia GitHub Push Setup Script
# =============================================================================
# Usage:
#   ./setup_github.sh <github_username>
#
# Example:
#   ./setup_github.sh thomas-hiddenpeak
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <github_username>"
    echo "Example: $0 thomas-hiddenpeak"
    exit 1
fi

GITHUB_USER="$1"
REPO_NAME="Resonantia"
REPO_URL="git@github.com:${GITHUB_USER}/${REPO_NAME}.git"
HTTPS_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

echo "=========================================="
echo " Resonantia GitHub Setup"
echo "=========================================="
echo ""

# Step 1: Check SSH key
echo "[1/5] Checking SSH key..."
if [ -f ~/.ssh/id_ed25519.pub ] || [ -f ~/.ssh/id_rsa.pub ]; then
    echo "  ✓ SSH key found"
else
    echo "  ✗ No SSH key found"
    echo ""
    echo "  Please create one first:"
    echo "    ssh-keygen -t ed25519 -C 'your_email@example.com'"
    echo "    Then add to GitHub: https://github.com/settings/keys"
    echo ""
    exit 1
fi

# Step 2: Test GitHub connection
echo "[2/5] Testing GitHub connection..."
if ssh -T git@github.com 2>&1 | grep -q "Hi \\|authenticated"; then
    echo "  ✓ GitHub SSH connection successful"
else
    echo "  ✗ GitHub SSH connection failed"
    echo "  Please verify your SSH key is added to GitHub"
    exit 1
fi

# Step 3: Configure git user (if not already set)
echo "[3/5] Configuring git user..."
if ! git config user.name &>/dev/null; then
    git config user.name "${GITHUB_USER}"
    echo "  ✓ Set user.name = ${GITHUB_USER}"
fi

if ! git config user.email &>/dev/null; then
    git config user.email "${GITHUB_USER}@users.noreply.github.com"
    echo "  ✓ Set user.email = ${GITHUB_USER}@users.noreply.github.com"
fi

# Step 4: Add remote
echo "[4/5] Adding remote repository..."
if git remote | grep -q origin; then
    CURRENT=$(git remote get-url origin)
    echo "  ⚠ Remote 'origin' already exists: ${CURRENT}"
    echo "  Updating to: ${REPO_URL}"
    git remote set-url origin "${REPO_URL}"
else
    git remote add origin "${REPO_URL}"
    echo "  ✓ Added remote 'origin': ${REPO_URL}"
fi

# Step 5: Show push commands
echo ""
echo "[5/5] Ready to push!"
echo ""
echo "=========================================="
echo " Next Steps:"
echo "=========================================="
echo ""
echo "1. Create the repository on GitHub first:"
echo "   ${HTTPS_URL}"
echo ""
echo "2. Then push with:"
echo "   git push -u origin main"
echo ""
echo "=========================================="

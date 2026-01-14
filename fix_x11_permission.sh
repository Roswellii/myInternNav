#!/bin/bash
# Fix X11 permission for Docker root user

echo "=== Fixing X11 Permission for Docker ==="
echo ""

# Method 1: Allow local root access (less secure but works)
echo "Method 1: Allowing local root access to X11..."
xhost +local:root

echo ""
echo "Current xhost permissions:"
xhost

echo ""
echo "✓ X11 permission fixed. Root user in Docker can now access X11."
echo ""
echo "Note: This is less secure. For production, use:"
echo "  xhost +SI:localuser:root"
echo ""
echo "To revert (remove root access):"
echo "  xhost -local:root"









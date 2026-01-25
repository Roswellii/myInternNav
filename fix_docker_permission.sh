#!/bin/bash
# 修复 Docker 权限问题
# 需要 sudo 权限执行

echo "正在修复 Docker 权限问题..."
echo ""

# 检查用户是否已经在 docker 组中
if groups | grep -q "\bdocker\b"; then
    echo "✓ 用户已经在 docker 组中"
    echo "如果仍然遇到权限问题，请尝试："
    echo "  1. 重新登录或运行: newgrp docker"
    echo "  2. 检查 Docker 服务是否运行: sudo systemctl status docker"
else
    echo "将当前用户 ($USER) 添加到 docker 组..."
    sudo usermod -aG docker $USER
    
    echo ""
    echo "✓ 用户已添加到 docker 组"
    echo ""
    echo "⚠️  重要提示："
    echo "  需要重新登录或运行以下命令使更改生效："
    echo "    newgrp docker"
    echo ""
    echo "或者直接重新登录系统"
fi

echo ""
echo "验证 Docker 权限（需要先执行 newgrp docker 或重新登录）："
echo "  运行: docker ps"
echo "  如果不再报错，说明权限已修复"































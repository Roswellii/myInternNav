#!/bin/bash
# 修复 DNS 以访问 conda 通道
# 需要 sudo 权限执行

echo "正在修复 DNS 配置..."

# 检查并添加 conda.anaconda.org
if ! grep -q "conda.anaconda.org" /etc/hosts 2>/dev/null; then
    echo "添加 conda.anaconda.org 到 /etc/hosts..."
    sudo bash -c 'echo "104.19.145.37 conda.anaconda.org" >> /etc/hosts'
    sudo bash -c 'echo "104.19.144.37 conda.anaconda.org" >> /etc/hosts'
fi

# 检查并添加 repo.anaconda.com
if ! grep -q "repo.anaconda.com" /etc/hosts 2>/dev/null; then
    echo "添加 repo.anaconda.com 到 /etc/hosts..."
    sudo bash -c 'echo "104.19.144.37 repo.anaconda.com" >> /etc/hosts'
    sudo bash -c 'echo "104.19.145.37 repo.anaconda.com" >> /etc/hosts'
fi

echo ""
echo "DNS 修复完成！"
echo "测试 DNS 解析..."
nslookup conda.anaconda.org 2>&1 | head -5 || echo "如果仍然失败，请检查网络连接"

























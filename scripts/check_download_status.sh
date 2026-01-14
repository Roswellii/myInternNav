#!/bin/bash
# 检查 InternVLA-N1-wo-dagger 模型下载状态

MODEL_DIR="/home/zhangwenqi/workspace/myInternNav/checkpoints/InternVLA-N1-wo-dagger"
LOG_FILE="${MODEL_DIR}/download.log"

echo "=========================================="
echo "InternVLA-N1-wo-dagger 下载状态"
echo "=========================================="
echo ""

# 检查下载进程
echo "📊 下载进程状态:"
if pgrep -f "continue_download_internvla_n1_wo_dagger.py" > /dev/null; then
    PID=$(pgrep -f "continue_download_internvla_n1_wo_dagger.py" | head -1)
    echo "  ✓ 下载进程正在运行 (PID: $PID)"
    ps -p $PID -o pid,pcpu,pmem,etime,cmd --no-headers | awk '{print "  CPU使用率: " $2 "%, 内存: " $3 "%, 运行时间: " $4}'
else
    echo "  ✗ 下载进程未运行"
fi
echo ""

# 检查权重文件
echo "📁 权重文件状态:"
for i in {1..4}; do
    FILE=$(printf "model-000%02d-of-00004.safetensors" $i)
    FILEPATH="${MODEL_DIR}/${FILE}"
    if [ -f "$FILEPATH" ]; then
        SIZE=$(du -h "$FILEPATH" | cut -f1)
        echo "  ✓ $FILE - $SIZE"
    else
        echo "  ✗ $FILE - 缺失"
    fi
done
echo ""

# 检查缓存目录大小
if [ -d "${MODEL_DIR}/.cache" ]; then
    CACHE_SIZE=$(du -sh "${MODEL_DIR}/.cache" 2>/dev/null | cut -f1)
    echo "💾 缓存目录大小: $CACHE_SIZE"
    echo ""
fi

# 显示日志最后几行
if [ -f "$LOG_FILE" ]; then
    echo "📝 最新日志 (最后10行):"
    echo "----------------------------------------"
    tail -10 "$LOG_FILE" | sed 's/^/  /'
    echo "----------------------------------------"
    echo ""
    echo "查看完整日志: tail -f $LOG_FILE"
fi

echo ""
echo "=========================================="













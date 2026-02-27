#!/bin/bash
# ============================================================
# Amazon目标域优化配置训练脚本
# ============================================================

echo "=========================================="
echo "Amazon目标域优化配置实验"
echo "=========================================="
echo ""
echo "优化配置 (基于网格搜索):"
echo "  - target_batch_size: 256 (默认64)"
echo "  - class_detect_alpha: 1 (默认3)"
echo "  - target_epochs: 10 (默认20)"
echo "  - lambda_pl: 0.5 (默认1.0)"
echo "  - lambda_proto: 0.05 (默认0.1)"
echo ""
echo "预期提升: 38% -> 50% (+12%)"
echo "=========================================="
echo ""

# ============================================================
# 实验1: DSLR -> Amazon (优化配置)
# ============================================================
echo "[$(date '+%H:%M:%S')] 开始: DSLR -> Amazon (优化配置)"

python main.py \
    --source dslr \
    --target amazon \
    --output_dir ./output_amazon_optimized \
    --seed 42 \
    --target_epochs 10 \
    --target_batch_size 256 \
    --buffer_capacity 300 \
    --class_detect_alpha 1 \
    --lambda_pl 0.5 \
    --lambda_proto 0.05 \
    --proto_temp 0.07 \
    --reliability_threshold 0.3

echo "[$(date '+%H:%M:%S')] 完成: DSLR -> Amazon (优化配置)"
echo ""

# ============================================================
# 实验2: Webcam -> Amazon (优化配置)
# ============================================================
echo "[$(date '+%H:%M:%S')] 开始: Webcam -> Amazon (优化配置)"

python main.py \
    --source webcam \
    --target amazon \
    --output_dir ./output_amazon_webcam_optimized \
    --seed 42 \
    --target_epochs 10 \
    --target_batch_size 256 \
    --buffer_capacity 300 \
    --class_detect_alpha 1 \
    --lambda_pl 0.5 \
    --lambda_proto 0.05

echo "[$(date '+%H:%M:%S')] 完成: Webcam -> Amazon (优化配置)"
echo ""

# ============================================================
# 显示结果摘要
# ============================================================
echo "=========================================="
echo "实验完成! 结果摘要:"
echo "=========================================="
echo ""

echo "DSL R -> Amazon (优化配置):"
tail -30 ./output_amazon_optimized/e3p_run.log | grep -E "(Final:|Peak:|Forgetting:)" || echo "  请查看日志"

echo ""
echo "Webcam -> Amazon (优化配置):"
tail -30 ./output_amazon_webcam_optimized/e3p_run.log | grep -E "(Final:|Peak:|Forgetting:)" || echo "  请查看日志"

echo ""
echo "=========================================="
echo "完整日志位置:"
echo "  DSLR->Amazon: ./output_amazon_optimized/e3p_run.log"
echo "  Webcam->Amazon: ./output_amazon_webcam_optimized/e3p_run.log"
echo "=========================================="

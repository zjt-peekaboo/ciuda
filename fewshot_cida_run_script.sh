#!/bin/bash

# Few-Shot CIDA 实验运行脚本
# 在Office-31数据集上运行不同few-shot设置的实验

GPU=0
SOURCE_EPOCHS=50
N_SHOT=5  

echo "======================================"
echo "Few-Shot CIDA Experiments on Office-31"
echo "======================================"
echo "GPU: $GPU"
echo "Source Epochs: $SOURCE_EPOCHS"
echo "Few-Shot Setting: $N_SHOT-shot"
echo ""

# 创建目录
mkdir -p data/office31
mkdir -p checkpoints
mkdir -p logs

run_experiment() {
    local source=$1
    local target=$2
    local n_shot=$3
    
    echo "--------------------------------------"
    echo "Running: $source -> $target ($n_shot-shot)"
    echo "--------------------------------------"
    
    # 根据n_shot调整参数
    if [ $n_shot -eq 1 ]; then
        target_epochs=80
        conf_threshold=0.98
        beta_supervised=3.0
        beta_prototype=1.5
    elif [ $n_shot -le 5 ]; then
        target_epochs=30
        conf_threshold=0.95
        beta_supervised=2.0
        beta_prototype=1.0
    else
        target_epochs=40
        conf_threshold=0.90
        beta_supervised=1.5
        beta_prototype=0.8
    fi
    
    python main.py \
        --source $source \
        --target $target \
        --mode both \
        --n_shot $n_shot \
        --source_epochs $SOURCE_EPOCHS \
        --target_epochs $target_epochs \
        --target_lr 0.0005 \
        --target_batch_size 16 \
        --unlabeled_batch_size 32 \
        --confidence_threshold $conf_threshold \
        --beta_supervised $beta_supervised \
        --beta_prototype $beta_prototype \
        --beta_pseudo 0.5 \
        --beta_consistency 0.3 \
        --strong_aug \
        --prioritize_labeled \
        --gpu $GPU \
        --seed 42
    
    echo ""
    echo "Completed: $source -> $target ($n_shot-shot)"
    echo ""
}


echo "======================================"
echo "Running experiments with different n-shot settings"
echo "======================================"


# for n_shot in 5 10; do

for n_shot in 5; do
    echo ""
    echo "======================================"
    echo "Testing $n_shot-shot setting"
    echo "======================================"
    
    # Amazon as source
    run_experiment "amazon" "webcam" $n_shot
    run_experiment "amazon" "dslr" $n_shot
    
    # DSLR as source
    run_experiment "dslr" "amazon" $n_shot
    run_experiment "dslr" "webcam" $n_shot
    
    # Webcam as source
    run_experiment "webcam" "amazon" $n_shot
    run_experiment "webcam" "dslr" $n_shot
    break
done

echo "======================================"
echo "All experiments completed!"
echo "======================================"
echo ""
echo "Results are saved in:"
echo "  - Checkpoints: ./checkpoints/"
echo "  - Logs: ./logs/"

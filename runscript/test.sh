#!/bin/bash

# 定义模型列表、字母列表和GBS列表
models=("1.3B")
#("2.1B" "4.7B" "6.2B")
letters=("m1")
gbs_values=(68)

for model in "${models[@]}"; do
    for gbs in "${gbs_values[@]}"; do
        for letter in "${letters[@]}"; do
            # 构造config文件名
            config="tangram_config_${model}-${letter}.json"
            
            # 构造日志文件和输出文件名
            log_file="logs/log_${model}_${letter}_${gbs}.log"
            throughput_file="logs/throughput_${model}_${letter}_${gbs}.txt"
            
            # 创建临时管道
            tmp_pipe=$(mktemp -u)
            mkfifo "$tmp_pipe"
            
            echo model: "$model", gbs: "$gbs", config: "$config"
            # 启动任务并同时记录日志
            stdbuf -oL bash run_test.sh "$model" "$gbs" "$config" > "$tmp_pipe" 2>&1 &
            pid=$!
            
            # 实时处理输出
            found_iteration6=false
            > "$throughput_file"  # 初始化输出文件
            
            tee "$log_file" < "$tmp_pipe" | while read -r line; do
                # 捕获throughput数据
                if [[ "$line" == *"throughput (sample/s): "* ]]; then
                    echo "$line" >> "$throughput_file"
                fi
                
                # 检查迭代6并终止
                if [[ "$line" == *"iteration        6"* ]]; then
                    found_iteration6=true
                    # 获取Slurm作业ID（新增部分）
                    jobid=$(squeue -u $USER -h | head -n1 | cut -d '+' -f1)
                    
                    if [ -n "$jobid" ]; then
                        echo "正在取消Slurm作业 $jobid..."
                        scancel $jobid 2>/dev/null
                    else
                        echo "警告：未找到对应的Slurm作业"
                    fi
                    kill -9 $pid 2>/dev/null
                    break
                fi
            done
            
            # 清理和等待
            wait $pid 2>/dev/null
            rm -f "$tmp_pipe"
            
            # 如果没有数据则创建空文件
            [ -s "$throughput_file" ] || touch "$throughput_file"
        done
    done
done
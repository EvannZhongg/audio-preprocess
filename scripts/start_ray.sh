#!/bin/bash
# ray start --head --port=6379 --object-store-memory=2147483648 --num-cpus=0
# ray start --address=11.177.169.159:6379 --object-store-memory=2147483648
# nohup python run_ray_task.py > /dev/null 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd $PROJECT_ROOT
pwd

# 定义头节点的IP地址
HEAD_NODE_IP="11.177.169.159"

# 检查参数数量
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {start-head|start-node|run-task|auto}"
    exit 1
fi

# 获取第一个参数作为操作类型
operation=$1

# 获取当前机器的IP地址
current_ip=$(hostname -I | awk '{print $1}')

case $operation in
    start-head)
        # 启动Ray head节点
        echo "Starting Ray head node"
        ray start --head --port=6379 --num-cpus=0
        ;;
    start-node)
        # 启动Ray worker节点
        echo "Starting Ray worker node"
        ray start --address=$HEAD_NODE_IP:6379
        ;;
    run-task)
        # 运行Ray任务
        echo "Running Ray task"
        nohup python run_ray_task.py > /dev/null 2>&1 &
        ;;
    auto)
        # 自动判断启动类型
        if [ "$current_ip" == "$HEAD_NODE_IP" ]; then
            echo "Detected head node IP, starting Ray head node"
            source activate AudioPipeline
            ray start --head --port=6379 --num-cpus=0
        else
            echo "Detected worker node IP, starting Ray worker node"
            source activate AudioPipeline
            ray start --address=$HEAD_NODE_IP:6379
        fi
        # 保持容器运行
        tail -f /dev/null
        ;;
    *)
        echo "Unknown operation: $operation"
        exit 1
        ;;
esac

# ray start --head --port=6379 --num-cpus=0
# ray start --address=11.177.169.159:6379
# nohup python run_ray_task.py > /dev/null 2>&1 &

#!/bin/bash

# 检查参数数量
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 {start-head|start-node|run-task}"
    exit 1
fi

# 获取第一个参数作为操作类型
operation=$1

case $operation in
    start-head)
        # 启动Ray head节点
        echo "Starting Ray head node"
        ray start --head --port=6379 --num-cpus=0
        ;;
    start-node)
        # 启动Ray worker节点
        echo "Starting Ray worker node"
        ray start --address=11.177.169.159:6379
        ;;
    run-task)
        # 运行Ray任务
        echo "Running Ray task"
        nohup python run_ray_task.py > /dev/null 2>&1 &
        ;;
    *)
        echo "Unknown operation: $operation"
        exit 1
        ;;
esac
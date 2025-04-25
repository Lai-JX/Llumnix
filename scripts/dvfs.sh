# 初始化
nvidia-smi -pl 300
nvidia-smi -pm ENABLED
nvidia-smi -acp 0

# 设置频率
nvidia-smi -lgc 2100
nvidia-smi -lmc 5000
nvidia-smi -ac 5000,2100

# 恢复
nvidia-smi -acp 1
nvidia-smi -rac 
nvidia-smi -rgc 
nvidia-smi -rmc
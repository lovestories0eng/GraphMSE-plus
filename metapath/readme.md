* 后台运行
```bash
nohup  --dataset IMDB --sample_times 4096 --sample_num 128 < /dev/null  > output.log 2>&1 &

# 脚本
nohup ./experiment.sh < /dev/null  > output.log 2>&1 &
```

sample_times: 每条类型各采样的条数
sample_num: 用于聚合的元路径数量
dataset: 数据集
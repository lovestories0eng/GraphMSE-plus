#!/bin/bash

# 定义一个参数数组，每行是一组参数
args_list=(
    "--dataset IMDB --sample_times 4096 --sample_num 2"
    "--dataset IMDB --sample_times 4096 --sample_num 4"
    "--dataset IMDB --sample_times 4096 --sample_num 8"
    "--dataset IMDB --sample_times 4096 --sample_num 16"
    "--dataset IMDB --sample_times 4096 --sample_num 32"
    "--dataset IMDB --sample_times 4096 --sample_num 64"
    "--dataset IMDB --sample_times 4096 --sample_num 128"
    "--dataset IMDB --sample_times 4096 --sample_num 256"
    "--dataset IMDB --sample_times 4096 --sample_num 512"
    "--dataset IMDB --sample_times 4096 --sample_num 1024"
)

# Python 文件名
PYTHON_FILE="main_bak.py"

# 循环执行 Python 文件，依次传入不同参数
for args in "${args_list[@]}"
do
    echo "Running: python $PYTHON_FILE $args"
    python "$PYTHON_FILE" $args
    echo "Finished running with args: $args"
    echo "--------------------------------"
done

echo "All runs completed."

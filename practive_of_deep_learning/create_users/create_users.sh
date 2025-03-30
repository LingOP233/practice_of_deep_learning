#!/bin/bash

# 检查 user_list.txt 是否存在
if [ ! -f "user_list.txt" ]; then
  echo "错误：user_list.txt 文件不存在" >&2
  exit 1
fi

# 读取user_list.txt中的每个用户名
while read username; do
    # 跳过空行
    [ -z "$username" ] && continue

    # 检查用户是否存在
    if id "$username" &>/dev/null; then
    	echo "$username以存在"
        continue
    else
        # 使用sudo创建用户（如果需要）
        if sudo useradd -m "$username" &>/dev/null; then
            echo "用户<$username>创建成功"
        else
            echo "错误：无法创建用户<$username>" >&2
        fi
    fi
done < user_list.txt

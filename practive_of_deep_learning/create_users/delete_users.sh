#!/bin/bash

# 检查 user_list.txt 是否存在
if [ ! -f "user_list.txt" ]; then
  echo "错误：user_list.txt 文件不存在" >&2
  exit 1
fi

# 逐行读取用户名并删除
while read username; do
  # 跳过空行
  [ -z "$username" ] && continue

  # 检查用户是否存在
  if id "$username" &>/dev/null; then
    # 使用 sudo 删除用户（包括家目录和邮件池）
    if sudo userdel -r "$username" &>/dev/null; then
      echo "用户<$username>删除成功"
    else
      echo "错误：无法删除用户<$username>" >&2
    fi
  else
    echo "提示：用户<$username>不存在，已跳过" >&2
  fi
done < user_list.txt

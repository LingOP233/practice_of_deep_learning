#!/bin/bash

{
    # 显示当前用户
    echo "当前用户：$(whoami)"
    
    # 显示当前系统时间（格式化为YYYY-MM-DD HH:MM:SS）
    echo -e "\n系统时间：$(date '+%Y-%m-%d %H:%M:%S')"
    
    # 显示磁盘使用情况
    echo -e "\n磁盘使用情况："
    df -h
    
    # 显示CPU负载情况（使用uptime提取负载信息）
    echo -e "\nCPU负载情况："
    uptime | awk -F 'load average: ' '{print $2}'
    
    # 显示内存使用情况
    echo -e "\n内存使用情况："
    free -h
} > system_report.txt

echo "系统报告已保存至 system_report.txt"

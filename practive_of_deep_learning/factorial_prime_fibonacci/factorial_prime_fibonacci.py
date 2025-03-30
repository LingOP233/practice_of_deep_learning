#!/usr/bin/env python3
import os
import sys
import math

def factorial(n):
    # 处理0的阶乘
    if n == 0:
        return 1
    
    result = 1
    # 使用循环计算阶乘
    for i in range(1, n + 1):
        result *= i
    return result

def is_prime(n):
    # 处理小于2的情况
    if n < 2:
        return False
    
    # 单独处理2（唯一的偶数素数）
    if n == 2:
        return True
    
    # 排除其他偶数
    if n % 2 == 0:
        return False
    
    # 只需检查奇数因子至平方根
    max_divisor = math.isqrt(n)
    for i in range(3, max_divisor + 1, 2):
        if n % i == 0:
            return False
    
    return True
def fibonacci(n):

    # 初始化基础数列
    sequence = []
    if n >= 1:
        sequence.append(0)
    if n >= 2:
        sequence.append(1)

    # 生成后续数列项
    for _ in range(2, n):  # 从第三项开始计算
        next_val = sequence[-1] + sequence[-2]
        sequence.append(next_val)
    
    return sequence

def get_menu_choice():
    """获取用户菜单选择（1-4）"""
    while True:
        print("\n" + "="*30)
        print("\n请选择要执行的功能：")
        print("1. 计算阶乘")
        print("2. 判断素数")
        print("3. 生成斐波那契数列")
        print("4. 退出程序\n")
        choice = input("请输入选项(1-4): ").strip()
        if choice in ['1', '2', '3', '4']:
            return int(choice)
        print("\n\033[91m⚠️ 输入无效,请输入1-4的数字!\033[0m")
def get_valid_input(prompt, check_type="non_negative"):
    """通用输入验证函数
    check_type参数:
    - 'non_negative'：需要非负整数（用于阶乘和斐波那契）
    - 'any_integer'：允许任意整数（用于素数判断）
    """
    while True:
        try:
            value_str = input(prompt).strip()
            
            # 尝试转换为数值类型
            if '.' in value_str:
                value = float(value_str)
                if not value.is_integer():
                    raise ValueError("必须为整数")
                value = int(value)
            else:
                value = int(value_str)
                
            # 根据类型检查要求验证
            if check_type == "non_negative" and value < 0:
                raise ValueError("必须为非负整数")
                
            return value
        except ValueError as e:
            print(f"\033[93m⚠️ 输入错误：{e}，请重新输入\033[0m")
def main():

 
    red ='\033[91m'
    yellow ='\033[93m'
    green ='\033[92m'
    blue ='\033[94m'
    reset ='\033[0m'
  

    print(f"{green}欢迎使用数学工具箱！{reset}")
    while True:
        choice = get_menu_choice()
        
        if choice == 1:  # 计算阶乘
            print("\n" + "-"*20)
            n = get_valid_input(f"{blue}请输入要计算阶乘的非负整数n: {reset}","non_negative")
            try:
                print(f"\n🔥 结果：{n}! = {factorial(n)}")
            except Exception as e:
                print(f"{red}❌ 计算错误：{str(e)}{reset}")
                
        elif choice == 2:  # 判断素数
            print("\n" + "-"*20)
            n = get_valid_input(f"{blue}请输入要判断的整数n: {reset}", "non_negative")
            try:
                result = is_prime(n)
                print(f"\n🔍 结果：{n} {'是' if result else '不是'}素数")
            except Exception as e:
                print(f"{red}❌ 判断错误：{str(e)}{reset}")
                
        elif choice == 3:  # 生成斐波那契数列
            print("\n" + "-"*20)
            n = get_valid_input(f"{blue}请输入要生成的数列长度n: {reset}", "non_negative")
            try:
                fib = fibonacci(n)
                print(f"\n📊 生成的斐波那契数列：{fib}")
            except Exception as e:
                print(f"{red}❌ 生成错误：{str(e)}{reset}")
                
        elif choice == 4:  # 退出程序
            print(f"{green}\n感谢使用,再见！{reset}\n")
            break
    
if __name__ == "__main__":
    main()
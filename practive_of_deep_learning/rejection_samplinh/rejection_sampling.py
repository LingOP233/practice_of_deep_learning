import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import time
import os
from colorama import init, Fore, Style
from tqdm import tqdm
from scipy.integrate import quad

# 初始化colorama
init(autoreset=True)



def rejection_sampling(target_pdf, proposal_pdf, proposal_sampler, M, n_samples):
    """
    拒绝采样方法生成符合目标分布的样本
    
    参数:
    - target_pdf: 目标概率密度函数
    - proposal_pdf: 建议分布的概率密度函数
    - proposal_sampler: 建议分布的采样函数
    - M: 建议分布与目标分布的比例上界
    - n_samples: 需要生成的样本数量
    
    返回:
    - 符合目标分布的样本
    - 接受率
    """
    samples = []
    total_iterations = 0
    
    try:
        # 创建进度条
        pbar = tqdm(total=n_samples, desc=f"{Fore.GREEN}生成样本{Style.RESET_ALL}", 
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]", 
                   ncols=100)
        
        while len(samples) < n_samples:
            # 从建议分布中采样
            x = proposal_sampler()
            # 计算接受概率
            proposal_val = proposal_pdf(x)
            # 检查除数是否为0，避免除零错误
            if proposal_val <= 0:
                accept_prob = 0  # 如果建议分布概率为0，则接受概率为0
            else:
                accept_prob = target_pdf(x) / (M * proposal_val)
            
            # 生成一个随机数，用于决定是否接受样本
            u = np.random.uniform(0, 1)
            
            # 先增加迭代次数，避免除零错误
            total_iterations += 1
            
            if u < accept_prob:
                samples.append(x)
                # 更新进度条
                pbar.update(1)
                # 更新进度条描述，显示当前接受率
                pbar.set_postfix({"接受率": f"{Fore.CYAN}{len(samples)/total_iterations:.4f}{Style.RESET_ALL}"})
            
        
        pbar.close()
        # 避免除零错误
        acceptance_rate = len(samples) / total_iterations if total_iterations > 0 else 0
        return np.array(samples), acceptance_rate
    except KeyboardInterrupt:
        # 确保进度条关闭
        pbar.close()
        print(f"\n\n{Fore.YELLOW}采样过程被用户中断{Style.RESET_ALL}")
        # 如果已经有样本，则返回当前样本，否则抛出异常
        if len(samples) > 0:
            acceptance_rate = len(samples) / total_iterations if total_iterations > 0 else 0
            print(f"{Fore.CYAN}返回已采集的 {len(samples)} 个样本{Style.RESET_ALL}")
            return np.array(samples), acceptance_rate
        raise
    except Exception as e:
        # 确保进度条关闭
        pbar.close()
        print(f"\n{Fore.RED}错误: 在采样过程中发生异常 - {str(e)}{Style.RESET_ALL}")
        raise

def plot_distribution(samples, target_pdf, x_range, title, bins=50):
    """
    绘制样本的直方图和目标概率密度函数
    
    参数:
    - samples: 生成的样本
    - target_pdf: 目标概率密度函数
    - x_range: x轴范围
    - title: 图表标题
    - bins: 直方图的箱数
    """
    try:
        # 设置中文字体
        try:
            font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")
            print(f"{Fore.CYAN}成功加载中文字体{Style.RESET_ALL}")
        except Exception as font_err:
            font = None
            print(f"{Fore.YELLOW}警告: 无法加载中文字体 - {str(font_err)}{Style.RESET_ALL}")
        
        plt.figure(figsize=(12, 6))
        
        # 计算统计分布
        hist_values, bins = np.histogram(samples, bins=bins, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_widths = bins[1:] - bins[:-1]
        # 计算误差
        errors = np.sqrt(hist_values * bin_widths / len(samples))
        # 绘制统计分布曲线
        plt.plot(bin_centers, hist_values, 'b-', label='统计分布曲线', linewidth=2)
        plt.fill_between(bin_centers, hist_values - errors, hist_values + errors, 
                         color='skyblue', alpha=0.3, label='统计误差范围')
        
        # 计算并绘制目标PDF
        x = np.linspace(x_range[0], x_range[1], 1000)
        y = [target_pdf(xi) for xi in x]
        plt.plot(x, y, 'r-', linewidth=2, label='目标概率密度函数')
        
        # 设置图表属性
        if font:
            plt.title(title, fontproperties=font, fontsize=16)
            plt.xlabel('银心距离 R (kpc)', fontproperties=font, fontsize=14)
            plt.ylabel('概率密度', fontproperties=font, fontsize=14)
            plt.legend(prop=font)
        else:
            plt.title(title, fontsize=16)
            plt.xlabel('银心距离 R (kpc)', fontsize=14)
            plt.ylabel('概率密度', fontsize=14)
            plt.legend()
        
        plt.grid(alpha=0.3)
        print(f"{Fore.GREEN}图表生成成功!{Style.RESET_ALL}")
        return hist_values, bins
    except Exception as e:
        print(f"{Fore.RED}错误: 在绘制分布图时发生异常 - {str(e)}{Style.RESET_ALL}")
        raise

def pulsar_density(r, R_sun=8.3, A=20.41, a=9.03, b=13.99, R_psr=3.76):
    """
    脉冲星密度分布函数(已归一化为概率密度函数)
    
    参数:
    - r: 距离银河系中心的距离 (kpc)
    - R_sun: 太阳-银河系中心距离 (kpc)
    - A: 密度系数 (kpc^-2)
    - a: 幂律指数
    - b: 指数衰减系数
    - R_psr: 特征距离 (kpc)
    
    返回:
    - 在距离r处的脉冲星概率密度
    """
    if r < 0:
        return 0
    
    ratio1 = (r + R_psr) / (R_sun + R_psr)
    ratio2 = (r - R_sun) / (R_sun + R_psr)
    
    # 计算未归一化的密度值
    unnormalized = A * (ratio1**a) * np.exp(-b * ratio2)
    
    # 归一化常数(使用另一个文件计算)

    normalization_constant = 0.002769866411707447  # 倒数
    
    return unnormalized * normalization_constant

def main():
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}\n")
    print(f"\n{Fore.CYAN}拒绝采样方法生成样本程序{Style.RESET_ALL}")
    
    # 预定义的一些目标分布
    distributions = {
        '1': {
            'name': '脉冲星密度分布(作业)',
            'pdf': pulsar_density,  # 使用定义好的函数
            'proposal': {
                'pdf':  lambda x: 1 / 30  if 0 <= x <= 30 else 0,# 均匀分布pdf
                'sampler': lambda: np.random.uniform(0, 30),  # 确保定义了采样函数
                'M': 1800  # 确保定义了M值
            },
            'range': (0, 30)  # 设定x_range的值
        },
    }
    # 主循环
    try:
        while True:
            # 显示可用分布
            print(f"{Fore.YELLOW}请选择目标分布:{Style.RESET_ALL}")
            print(f"{Fore.WHITE}0. {Fore.CYAN}退出程序{Style.RESET_ALL}")
            for key, dist in distributions.items():
                if 'name' in dist:
                    print(f"{Fore.WHITE}{key}. {Fore.CYAN}{dist['name']}{Style.RESET_ALL}")
            print()
            
            try:
                choice = input(f"{Fore.YELLOW}请输入选择 (0-1): {Style.RESET_ALL}")
                
                # 退出程序选项
                if choice == '0':
                    print(f"\n{Fore.CYAN}感谢使用，再见！{Style.RESET_ALL}")
                    return
                    
               
                else:
                    if choice not in distributions:
                        print(f"\n{Fore.RED}错误: 无效的选择 '{choice}'，请输入0-1之间的数字{Style.RESET_ALL}")
                        continue
                    dist_info = distributions[choice]
                # 获取样本数量（单位：万）
                try:
                    n_samples_in_ten_thousands = float(input(f"\n{Fore.YELLOW}请输入需要生成的样本数量(单位:万): {Style.RESET_ALL}"))
                    if n_samples_in_ten_thousands <= 0:
                        print(f"\n{Fore.RED}错误: 样本数量必须大于0{Style.RESET_ALL}")
                        continue
                    n_samples = int(n_samples_in_ten_thousands * 10000)
                
                    print(f"\n{Fore.CYAN}开始生成 {n_samples} 个符合{dist_info['name']}的样本...{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
                    start_time = time.time()
                    
                    # 执行拒绝采样
                    samples, acceptance_rate = rejection_sampling(
                        dist_info['pdf'],
                        dist_info['proposal']['pdf'],
                        dist_info['proposal']['sampler'],
                        dist_info['proposal']['M'],
                        n_samples
                    )
                    
                    end_time = time.time()
                    
                    print(f"\n{Fore.GREEN}样本生成完成!{Style.RESET_ALL}")
                    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
                    print(f"\n{Fore.WHITE}总样本数: {Fore.GREEN}{len(samples)}{Style.RESET_ALL}")
                    print(f"{Fore.WHITE}接受率: {Fore.GREEN}{acceptance_rate:.4f}{Style.RESET_ALL}")
                    print(f"{Fore.WHITE}耗时: {Fore.GREEN}{end_time - start_time:.2f} 秒{Style.RESET_ALL}\n")
            
                    # 绘制分布图
                    print(f"{Fore.CYAN}正在绘制分布图...{Style.RESET_ALL}")
                    hist_values, bins = plot_distribution(
                        samples, 
                        dist_info['pdf'], 
                        dist_info['range'],  # 使用定义的range
                        f"{dist_info['name']} - 样本量: {n_samples_in_ten_thousands}万"
                    )
                    
                    # 保存图像
                    # 获取脚本所在目录
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    filename = f"rejection_sampling_{dist_info['name']}_{n_samples}.png"
                    # 构建完整的文件路径
                    file_path = os.path.join(script_dir, filename)
                    plt.savefig(file_path, dpi=300, bbox_inches='tight')
                    print(f"{Fore.GREEN}图像已保存为: {file_path}{Style.RESET_ALL}")
                    
                    # 显示图像
                    plt.show()
            
                    # 询问是否保存样本数据
                    save_option = input(f"\n{Fore.YELLOW}是否保存样本数据到文件? (y/n): {Style.RESET_ALL}")
                    if save_option.lower() == 'y':
                        try:
                            # 获取脚本所在目录（如果前面没有定义）
                            if 'script_dir' not in locals():
                                script_dir = os.path.dirname(os.path.abspath(__file__))
                            filename = f"samples_{dist_info['name']}_{n_samples}.csv"
                            # 构建完整的文件路径
                            file_path = os.path.join(script_dir, filename)
                            np.savetxt(file_path, samples, delimiter=',')
                            print(f"\n{Fore.GREEN}样本数据已保存到: {file_path}{Style.RESET_ALL}")
                        except Exception as e:
                            print(f"\n{Fore.RED}错误: 保存样本数据时发生异常 - {str(e)}{Style.RESET_ALL}")
                    
                    # 询问是否继续
                    continue_option = input(f"\n{Fore.YELLOW}是否继续进行新的采样? (y/n): {Style.RESET_ALL}")
                    if continue_option.lower() != 'y':
                        print(f"\n{Fore.CYAN}感谢使用，再见！{Style.RESET_ALL}")
                        return
                except ValueError as ve:
                    print(f"\n{Fore.RED}错误: 输入的数值格式不正确 - {str(ve)}{Style.RESET_ALL}")
                    continue
                except Exception as e:
                    print(f"\n{Fore.RED}错误: {str(e)}{Style.RESET_ALL}")
                    continue
            except KeyboardInterrupt:
                # 内部循环中捕获Ctrl+C，返回到主菜单
                print(f"\n\n{Fore.YELLOW}操作被用户中断，返回主菜单{Style.RESET_ALL}")
                continue
            except Exception as e:
                print(f"\n{Fore.RED}错误: {str(e)}{Style.RESET_ALL}")
                continue
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}程序被用户中断{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}感谢使用，再见！{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}发生未预期的错误: {str(e)}{Style.RESET_ALL}")
        print(f"\n{Fore.CYAN}程序已终止{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        main()
        print(f"\n{Fore.CYAN}程序执行完毕，感谢使用！{Style.RESET_ALL}\n")
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}程序被用户中断，正在退出...{Style.RESET_ALL}\n")
    except Exception as e:
        print(f"\n{Fore.RED}程序执行过程中发生错误: {str(e)}{Style.RESET_ALL}\n")
    finally:
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")
        
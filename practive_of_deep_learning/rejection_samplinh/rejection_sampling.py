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
        
        # 优化：批量处理样本以提高性能
        batch_size = min(10000, n_samples // 10 + 1)  # 批量大小
        
        while len(samples) < n_samples:
            # 批量从建议分布中采样
            batch_x = np.array([proposal_sampler() for _ in range(batch_size)])
            
            # 批量计算接受概率
            proposal_vals = np.array([proposal_pdf(x) for x in batch_x])
            
            # 避免除零错误
            valid_indices = proposal_vals > 0
            accept_probs = np.zeros(batch_size)
            
            if np.any(valid_indices):
                # 向量化计算接受概率
                valid_x = batch_x[valid_indices]
                valid_proposal_vals = proposal_vals[valid_indices]
                target_vals = np.array([target_pdf(x) for x in valid_x])
                
                accept_probs[valid_indices] = target_vals / (M * valid_proposal_vals)
                
            # 生成随机数，用于决定是否接受样本
            u = np.random.uniform(0, 1, batch_size)
            
            # 选择接受的样本
            accepted_indices = u < accept_probs
            accepted_samples = batch_x[accepted_indices]
            
            # 更新计数
            total_iterations += batch_size
            
            # 添加接受的样本
            samples.extend(accepted_samples)
            
            # 避免样本过多
            if len(samples) > n_samples:
                samples = samples[:n_samples]
                
            # 更新进度条
            current_len = len(samples)
            pbar.n = current_len
            pbar.set_postfix({"接受率": f"{Fore.CYAN}{current_len/total_iterations:.4f}{Style.RESET_ALL}"})
            pbar.refresh()
            
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

def calculate_normalization_constant(pdf_func, x_min=0, x_max=30):
    """
    计算概率密度函数的归一化常数
    
    参数:
    - pdf_func: 概率密度函数
    - x_min: 积分下限
    - x_max: 积分上限
    
    返回:
    - 归一化常数的倒数
    """
    try:
        # 数值积分
        integral, _ = quad(pdf_func, x_min, x_max)
        if integral <= 0:
            raise ValueError("积分结果小于或等于零，无法归一化")
        return 1.0 / integral
    except Exception as e:
        print(f"{Fore.RED}计算归一化常数时出错: {str(e)}{Style.RESET_ALL}")
        return 0.002769866411707447  # 使用默认值作为备用

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
            # 改进字体检测，支持多种操作系统
            if os.name == 'nt':  # Windows
                font_path = r"C:\Windows\Fonts\simhei.ttf"
            else:  # Linux/Mac
                possible_paths = [
                    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                    "/System/Library/Fonts/PingFang.ttc"  # macOS
                ]
                font_path = next((p for p in possible_paths if os.path.exists(p)), None)
                
            if font_path and os.path.exists(font_path):
                font = FontProperties(fname=font_path)
                print(f"{Fore.CYAN}成功加载中文字体{Style.RESET_ALL}")
            else:
                font = None
                print(f"{Fore.YELLOW}警告: 找不到中文字体文件{Style.RESET_ALL}")
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
        y = np.array([target_pdf(xi) for xi in x])  # 使用numpy数组提高效率
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

def pulsar_density(r, R_sun=8.3, A=20.41, a=9.03, b=13.99, R_psr=3.76, prevent_recursion=False):
    """
    脉冲星密度分布函数(已归一化为概率密度函数)
    
    参数:
    - r: 距离银河系中心的距离 (kpc)
    - R_sun: 太阳-银河系中心距离 (kpc)
    - A: 密度系数 (kpc^-2)
    - a: 幂律指数
    - b: 指数衰减系数
    - R_psr: 特征距离 (kpc)
    - prevent_recursion: 防止递归调用
    
    返回:
    - 在距离r处的脉冲星概率密度
    """
    if r < 0:
        return 0
    
    ratio1 = (r + R_psr) / (R_sun + R_psr)
    ratio2 = (r - R_sun) / (R_sun + R_psr)
    
    # 计算未归一化的密度值
    unnormalized = A * (ratio1**a) * np.exp(-b * ratio2)
    
    # 如果是为了防止递归或已有归一化常数，则直接返回未归一化值
    if prevent_recursion:
        return unnormalized
    
    # 使用函数动态计算归一化常数，如果失败则使用预设值
    norm_const = getattr(pulsar_density, '_norm_const', None)
    if norm_const is None:
        # 定义未归一化的函数用于积分
        def unnormalized_pdf(x):
            if x < 0:
                return 0
            
            # 直接计算，避免递归调用pulsar_density
            ratio1 = (x + R_psr) / (R_sun + R_psr)
            ratio2 = (x - R_sun) / (R_sun + R_psr)
            return A * (ratio1**a) * np.exp(-b * ratio2)
        
        # 计算归一化常数
        norm_const = calculate_normalization_constant(unnormalized_pdf)
        pulsar_density._norm_const = norm_const
    
    return unnormalized * norm_const

def create_output_directory():
    """创建输出目录"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"{Fore.GREEN}已创建输出目录: {output_dir}{Style.RESET_ALL}")
        return output_dir
    except Exception as e:
        print(f"{Fore.YELLOW}创建输出目录时发生错误: {str(e)}{Style.RESET_ALL}")
        return os.path.dirname(os.path.abspath(__file__))

def main():
    print(f"{Fore.CYAN}{'=' * 50}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}拒绝采样方法生成样本程序{Style.RESET_ALL}")
    
    # 创建输出目录
    output_dir = create_output_directory()
    
    # 预定义的一些目标分布
    distributions = {
        '1': {
            'name': '脉冲星密度分布',
            'pdf': pulsar_density,  # 使用定义好的函数
            'proposal': {
                'pdf': lambda x: 1 / 30 if 0 <= x <= 30 else 0,  # 均匀分布pdf
                'sampler': lambda: np.random.uniform(0, 30),  # 均匀分布采样函数
                'M': 4.8  # 确保定义了M值
            },
            'range': (0, 30)  # 设定x_range的值
        },
    }
    
    # 主循环
    try:
        while True:
            # 显示可用分布
            print(f"\n{Fore.YELLOW}请选择目标分布:{Style.RESET_ALL}")
            print(f"{Fore.WHITE}0. {Fore.CYAN}退出程序{Style.RESET_ALL}")
            for key, dist in distributions.items():
                if 'name' in dist:
                    print(f"{Fore.WHITE}{key}. {Fore.CYAN}{dist['name']}{Style.RESET_ALL}")
            print()
            
            try:
                choice = input(f"{Fore.YELLOW}请输入选择 (0-{len(distributions)}): {Style.RESET_ALL}")
                
                # 退出程序选项
                if choice == '0':
                    print(f"\n{Fore.CYAN}感谢使用，再见！{Style.RESET_ALL}")
                    return
                    
                else:
                    if choice not in distributions:
                        print(f"\n{Fore.RED}错误: 无效的选择 '{choice}'，请输入0-{len(distributions)}之间的数字{Style.RESET_ALL}")
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
                    print(f"{Fore.WHITE}总样本数: {Fore.GREEN}{len(samples)}{Style.RESET_ALL}")
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
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"rejection_sampling_{dist_info['name']}_{n_samples}_{timestamp}.png"
                    # 构建完整的文件路径
                    file_path = os.path.join(output_dir, filename.replace(" ", "_").replace("(", "").replace(")", ""))
                    plt.savefig(file_path, dpi=300, bbox_inches='tight')
                    print(f"{Fore.GREEN}图像已保存为: {file_path}{Style.RESET_ALL}")
                    
                    # 显示图像
                    plt.show()
            
                    # 询问是否保存样本数据
                    save_option = input(f"\n{Fore.YELLOW}是否保存样本数据到文件? (y/n): {Style.RESET_ALL}")
                    if save_option.lower() == 'y':
                        try:
                            filename = f"samples_{dist_info['name']}_{n_samples}_{timestamp}.csv"
                            # 构建完整的文件路径
                            file_path = os.path.join(output_dir, filename.replace(" ", "_").replace("(", "").replace(")", ""))
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
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from matplotlib.colors import ListedColormap
import time

# 设置随机种子和生成数据
use_custom_seed = input("是否使用自定义种子？(选择否则使用程序生成的随机种子)   (y/n): ").lower() == 'y'
if use_custom_seed:
    try:
        random_seed = int(input("请输入自定义种子(整数): "))
        if random_seed < 0 or random_seed > 2**32-1:
            raise ValueError("种子值必须在0到2^32-1之间")
    except ValueError as e:
        print(f"输入错误: {e}")
        use_random_seed = input("是否使用程序生成的随机种子继续？(y/n): ").lower() == 'y'
        if use_random_seed:
            random_seed = int.from_bytes(os.urandom(4), byteorder="big")
            print(f"使用程序生成的随机种子: {random_seed}")
        else:
            print("程序退出")
            sys.exit(0)
else:
    random_seed = int.from_bytes(os.urandom(4), byteorder="big")
    print(f"使用程序生成的随机种子: {random_seed}")

np.random.seed(random_seed)                             #图方便把时间作为随机种子
X, y = make_moons(200, noise=0.20)
# 训练集大小
num_examples = len(X)

# 输入层维度（二维坐标输入）
nn_input_dim = 2

# 输出层维度（2个类别，使用 one-hot 编码）
nn_output_dim = 2

# 梯度下降参数（手动选择的超参数）
epsilon = 0.01        # 学习率（learning rate）
reg_lambda = 0.01     # 正则化强度（L2 正则项系数）


def plot_decision_boundary(pred_func, ax=None):
    # 设置边界范围和网格间隔
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 网格点的预测
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 如果没有提供ax，则使用当前活动的轴
    if ax is None:
        ax = plt.gca()
        
    # ===== 自定义填充颜色 =====
    cmap_background = ListedColormap(['#a0c4ff', '#ffc9c9'])  # 浅蓝+浅橙
    ax.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.6)

    # 自定义颜色和形状
    colors = ['CornflowerBlue', 'Tomato']     # 蓝、橙
    markers = ['o', '*']                # o: 圆形, *: 星形

    for i in range(2):  # 类别 0 和 1
        ax.scatter(X[y==i, 0], X[y==i, 1],
                    s=50,   #点的大小
                    c=colors[i],  #点的颜色
                    marker=markers[i],  #点的形状
                    label=f'Class {i}',  #标签
                    edgecolors='None',  #点的边缘颜色
                    alpha=0.8)  #点的透明度
        
# 训练神经网络，学习模型参数并返回最终模型
# 参数说明：
# - nn_hdim：隐藏层的节点数量
# - num_passes：迭代次数（训练轮数）
# - print_loss：是否每 1000 次打印一次损失
def build_model(nn_hdim, num_passes=30000, print_loss=False):
    np.random.seed(0)

    # 参数初始化（权重随机初始化 + 偏置初始化为 0）
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    model = {}
    
    print(f"\n开始训练隐藏层大小为 {nn_hdim} 的模型...")
    update_interval = max(1, num_passes // 20)  # 更新进度条的频率
    
    # 训练过程：使用全量批量梯度下降（Batch GD）
    for i in range(num_passes):
        # -------- 前向传播 --------
        z1 = X.dot(W1) + b1               # 输入层 → 隐藏层
        a1 = 1 / (1 + np.exp(-z1))        # 激活函数：sigmoid
        z2 = a1.dot(W2) + b2              # 隐藏层 → 输出层
        exp_scores = np.exp(z2)           # softmax 的分子部分
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # softmax 概率

        # -------- 反向传播 --------
        delta3 = probs
        delta3[range(num_examples), y] -= 1         # 输出误差（预测 - 真实）交叉熵损失函数与softmax结合
        dW2 = a1.T.dot(delta3)                      # 输出层权重梯度
        db2 = np.sum(delta3, axis=0, keepdims=True) # 输出层偏置梯度

        delta2 = delta3.dot(W2.T) * a1 * (1 - a1)  # sigmoid的导数：a1 * (1 - a1)
        dW1 = X.T.dot(delta2)                      # 隐藏层权重梯度
        db1 = np.sum(delta2, axis=0)               # 隐藏层偏置梯度

        # -------- 正则化（L2）--------
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # -------- 参数更新（梯度下降）--------
        W1 -= epsilon * dW1
        b1 -= epsilon * db1
        W2 -= epsilon * dW2
        b2 -= epsilon * db2

        # 保存更新后的参数
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # 显示进度条和损失
        if i % update_interval == 0 or i == num_passes - 1:
            progress = int((i + 1) / num_passes * 50)  # 50是进度条的长度
            progress_bar = "[" + "=" * progress + " " * (50 - progress) + "]"
            progress_percentage = (i + 1) / num_passes * 100
            
            if print_loss:
                current_loss = calculate_loss(model)
                sys.stdout.write(f"\r{progress_bar} {progress_percentage:.1f}% 损失: {current_loss:.6f}")
            else:
                sys.stdout.write(f"\r{progress_bar} {progress_percentage:.1f}%")
            
            sys.stdout.flush()
    
    print("\n训练完成!")
    return model
# 计算整个数据集上的总损失（用于评估模型效果）
def calculate_loss(model):
    # 从模型中提取参数
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']

    # 前向传播，计算预测概率
    z1 = X.dot(W1) + b1                   # 输入层 → 隐藏层
    a1 = 1 / (1 + np.exp(-z1))             # 激活函数：sigmoid
    z2 = a1.dot(W2) + b2                  # 隐藏层 → 输出层
    exp_scores = np.exp(z2)               # 对每个类别计算 e^score
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # softmax 概率分布

    # 计算交叉熵损失（对数损失）
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)

    # 加入 L2 正则化项（防止过拟合）
    data_loss += (reg_lambda / 2) * (
        np.sum(np.square(W1)) + np.sum(np.square(W2))
    )

    # 返回平均损失
    return data_loss / num_examples
# 预测函数：根据输入样本 x，输出类别（0 或 1）
def predict(model, x):
    # 解包模型参数
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']

    # 前向传播，计算每个类别的概率
    z1 = x.dot(W1) + b1          # 输入层 → 隐藏层
    a1 = 1 / (1 + np.exp(-z1))   # 激活函数：sigmoid
    z2 = a1.dot(W2) + b2         # 隐藏层 → 输出层
    exp_scores = np.exp(z2)      # 指数函数
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # softmax 概率分布

    # 返回每个样本概率最大的类别索引（即预测结果）
    return np.argmax(probs, axis=1)

def main():
    # 可视化不同隐藏层节点数对模型决策边界的影响
    plt.figure(figsize=(14, 12))  # 调整整体图像大小为更合适的比例

    # 隐藏层节点数量列表
    hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]
    
    print("将训练7个不同隐藏层大小的模型，请耐心等待...")

    # 设置子图布局：4行2列
    for i, nn_hdim in enumerate(hidden_layer_dimensions):
        ax = plt.subplot(4, 2, i + 1)  # 创建子图：4行2列而不是5行2列
        plt.title(f"Hidden Layer size: {nn_hdim}", fontsize=12)  # 设置子图标题
        model = build_model(nn_hdim, print_loss=True)  # 训练模型，显示损失
        plot_decision_boundary(lambda x: predict(model, x), ax=ax)  # 使用当前子图绘制决策边界
        plt.xlabel("X1", fontsize=10)
        plt.ylabel("X2", fontsize=10)
        if i == 0:  # 只在第一个子图显示图例
            plt.legend(fontsize=8)

    # 调整子图间距
    plt.subplots_adjust(hspace=0.3, wspace=0.3)  # 增加子图之间的水平和垂直间距

    # 显示所有子图
    plt.tight_layout()
    
    # 确保保存图片
    try:
        # 获取当前脚本所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 构建图片保存路径
        save_path = os.path.join(current_dir, "ai_net_img.png")
        # 保存图片
        plt.savefig(save_path, dpi=300)
        print(f"图片已保存至: {save_path}")
    except Exception as e:
        print(f"保存图片失败: {e}")
        print("尝试保存到当前工作目录...")
        try:
            # 尝试保存到当前工作目录
            plt.savefig("ai_net_img_04.png", dpi=300)
            print(f"图片已保存至当前工作目录: {os.getcwd()}/ai_net_img.png")
        except Exception as e2:
            print(f"再次保存失败: {e2}")
    
    # 显示图像
    plt.show()

if __name__ == "__main__":
    main()

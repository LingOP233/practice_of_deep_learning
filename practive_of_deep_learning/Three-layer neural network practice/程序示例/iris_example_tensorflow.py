#!/usr/bin/env python

import sys
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 加载数据集
iris = load_iris()

print(iris.data.shape)
#sys.exit(1)
# 构建 DataFrame 方便可视化
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].apply(lambda x: iris.target_names[x])

# 取前10个样本
sample_df = df.head(10)

# 设置样式
sns.set(style="whitegrid")

# 画条形图：每个样本的每个特征
sample_df_plot = sample_df.drop(columns=['target'])
sample_df_plot.index = [f"Sample {i}" for i in range(10)]

# 转置以便用条形图绘制
transposed = sample_df_plot.drop(columns=["species"]).T

# 绘图
plt.figure(figsize=(12, 6))
transposed.plot(kind="bar", figsize=(14, 6), colormap="tab10")
plt.title("Iris Dataset - First 10 Samples Feature Overview")
plt.ylabel("Value")
plt.xlabel("Features")
plt.xticks(rotation=0)
plt.legend(title="Sample Index", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# 预处理数据
X = iris.data
y = iris.target

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# one-hot 编码标签
y_cat = tf.keras.utils.to_categorical(y, num_classes=3)

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# 2. 构建模型
#model = tf.keras.models.Sequential([
#    tf.keras.layers.Dense(80, activation='relu', input_shape=(4,)),  # 隐藏层
#    tf.keras.layers.Dense(3, activation='softmax')  # 输出层，使用 softmax
#])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(84, activation='relu', input_shape=(4,)),   # 第一隐藏层
    tf.keras.layers.Dense(32, activation='tanh'),                     # 第二隐藏层，激活函数换成 tanh
    tf.keras.layers.Dense(16, activation='elu'),                      # 第三隐藏层，使用 ELU
    tf.keras.layers.Dense(3, activation='softmax')                    # 输出层，分类用 softmax
])

# 3. 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # 使用 categorical_crossentropy，需 one-hot 编码
              metrics=['accuracy'])

# 4. 训练模型
# epochs=15 模型将遍历整个训练集的次数（共训练15轮 epochs）
# verbose=1 控制训练过程的打印信息：
#  0 = 不输出
#  1 = 每轮输出一个进度条
#  2 = 每轮输出一行
model.fit(X_train, y_train, epochs=15, batch_size=26, verbose=1)

# 5. 评估模型
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"测试准确率: {accuracy:.2f}")

import numpy as np

sample = np.array([X_test[0]])  # 随便取一个测试样本
probs = model.predict(sample)
print("预测概率分布:", probs)
print("预测类别:", np.argmax(probs))
print(y_test[0])

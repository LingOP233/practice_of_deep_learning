from scipy.integrate import quad
import numpy as np

# 定义未归一化的原始函数
def original_pulsar_density(r):
    R_sun = 8.3
    A = 20.41
    a = 9.03 
    b = 13.99
    R_psr = 3.76
    
    if r < 0:
        return 0
    
    ratio1 = (r + R_psr) / (R_sun + R_psr)
    ratio2 = (r - R_sun) / (R_sun + R_psr)
    
    return A * (ratio1**a) * np.exp(-b * ratio2)

# 计算归一化常数
integral, _ = quad(original_pulsar_density, 0, 30)
normalization_constant = 1.0 / integral
print(f"归一化常数: {normalization_constant}")
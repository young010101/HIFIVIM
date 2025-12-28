# %%
import numpy as np
import matplotlib.pyplot as plt

N = 256
K = 10
M = 32

np.random.seed(42)
x = np.zeros(N)
positions = np.random.choice(N, K, replace=False)
x[positions] = np.random.randn(K) * 10
# %%生成随机测量矩阵 phi （高斯随机矩阵）
Phi = np.random.randn(M, N) / np.sqrt(M)
# %%
y = Phi @ x

# %%
from scipy.linalg import lstsq

x_rec = np.zeros(N)
residual = y.copy()
support = [] # 已选的索引

for _ in range(K+5):
    correlations = Phi.T @ residual
    atom_idx = np.argmax(np.abs(correlations))
    if atom_idx in support:
        break
    support.append(atom_idx)
    
    Phi_s = Phi[:, support]
    s = lstsq(Phi_s, y)[0]
    
    residual = y - Phi_s @ s
    
x_rec[support] = s
# %%
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.stem(x)
plt.subplot(2,1,2)
plt.stem(x_rec)

# %% 计算重构误差
reconstruction_error = np.linalg.norm(x - x_rec) / np.linalg.norm(x)
print(reconstruction_error)


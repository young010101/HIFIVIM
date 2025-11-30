# %%
import numpy as np
import matplotlib.pyplot as plt

N = 256
K = 10
M = 32

np.random.seed(42)
x = np.zeros(N)
pos = np.random.choice(N, K, replace=None)
x[pos] = np.random.randn(K) * 10

Phi = np.random.randn(M, N)
y = Phi @ x

# %% method 1
x_rec1 = np.linalg.pinv(Phi) @ y

# %% method 2
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01, max_iter=10000)
lasso.fit(Phi, y)
x_rec2 = lasso.coef_

# %% method 3
from scipy.linalg import lstsq
x_rec3 = np.zeros(N)
residual = y.copy()
support = []

for _ in range(K + 5):
    correlations = Phi.T @ residual
    atom_max = np.argmax(np.abs(correlations))
    if atom_max in support:
        break
    support.append(atom_max)
    Phi_s = Phi[:, support]
    s = lstsq(Phi_s, y)[0]
    residual = y - Phi_s @ s
    
x_rec3[support] = s
    

# %% 
plt.figure(figsize=(15,12))
plt.subplot(411)
plt.stem(x)
plt.subplot(412)
plt.stem(x_rec1)
plt.subplot(413)
plt.stem(x_rec2)
plt.subplot(414)
plt.stem(x_rec3)
#%% 测试 1: IVIM 模型基础
import sys
import os

# 添加 code 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 导入主要模块
from utils import ivim_model, phase_estimate_
from Phantom_utils import calc_rmse, get_other_metrics

import matplotlib.pyplot as plt

print("=" * 50)
print("测试 1: IVIM 模型")
print("=" * 50)

# 定义 b 值（扩散权重）
bvals = [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]

# IVIM 参数示例（白质参数）
f = 0.07      # 灌注分数 (perfusion fraction)
D = 0.0006    # 扩散系数 (ADC) mm²/s
Dstar = 0.045 # 伪扩散系数 (pseudodiffusion) mm²/s

# 生成 IVIM 信号
signal = ivim_model(f, D, Dstar, bvals)

print(f"\n参数设置:")
print(f"  灌注分数 f = {f}")
print(f"  扩散系数 D = {D} mm²/s")
print(f"  伪扩散系数 D* = {Dstar} mm²/s")
print(f"\n生成的信号 (归一化):")
print(f"  b值: {bvals}")
print(f"  信号: {[f'{s:.4f}' for s in signal[:5]]}... (前5个)")

# 可视化信号衰减曲线
plt.figure(figsize=(10, 6))
plt.plot(bvals, signal, 'o-', linewidth=2, markersize=8)
plt.xlabel('b-value (s/mm²)', fontsize=12, fontweight='bold')
plt.ylabel('Normalized Signal', fontsize=12, fontweight='bold')
plt.title('IVIM Signal Decay Curve', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sim_diff/ivim_signal_curve.png', dpi=150, bbox_inches='tight')
print(f"\n✓ 信号曲线已保存到: sim_diff/ivim_signal_curve.png")
# %%

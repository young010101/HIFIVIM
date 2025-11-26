# HIFIVIM 库学习笔记

> 学习日期：2025-11-26
> 项目：High-Fidelity Intravoxel Parameter Estimation Using Locally Low Rank and Subspace Modeling

## 📚 项目概览

**HIFIVIM** 是一个用于 **高保真度体素内参数估计** 的 Python 库，基于局部低秩（Locally Low Rank）和子空间建模技术。这个项目对应的学术论文是关于 IVIM（Intravoxel Incoherent Motion，体素内非相干运动）参数映射的研究。

### 🎯 主要应用场景
- **医学影像分析**：特别是磁共振成像（MRI）
- **IVIM 参数估计**：用于分析组织的灌注和扩散特性
- **脑部影像研究**：支持数值脑phantom分析

### 📄 相关论文
- **标题**：High-Fidelity Intravoxel Incoherent Motion Parameter Mapping Using Locally Low-Rank and Subspace Modeling
- **作者**：Finkelstein, A. J., Liao, C., Cao, X., Mani, M., Schifitto, G., & Zhong, J.
- **期刊**：NeuroImage (2024)
- **链接**：https://www.sciencedirect.com/science/article/pii/S105381192400096X

## 🏗️ 项目结构

```
HIFIVIM/
├── README.md              # 项目说明文档
├── code/                  # 核心代码目录
│   ├── dictionary_gen.py  # IVIM字典生成和基函数提取
│   ├── Phantom_IVIM.py    # 数值脑phantom处理
│   ├── Phantom_utils.py   # Phantom相关工具函数
│   ├── IVIM_LLR.py        # 主重建算法（LLR+子空间）
│   ├── utils.py           # 核心工具函数库
│   └── cfl.py             # BART格式文件I/O
├── Phantom/               # phantom数据文件
│   ├── phase_estimates.nii.gz
│   └── sens_maps.nii.gz
└── images/
    └── GraphicalAbstract.png  # 算法流程图
```

## 🛠️ 环境要求

### 必需软件
- **Python 3.9**
- **BART v1.09** (Berkeley Advanced Reconstruction Toolbox)
- **MATLAB 2021a** (部分功能需要)

### 主要Python依赖
```python
numpy           # 数值计算
matplotlib      # 数据可视化
scipy          # 科学计算
scikit-image   # 图像处理
dipy           # 扩散MRI分析
emcee          # 贝叶斯采样
corner         # 参数分布可视化
multiprocessing # 并行计算
```

## 🔧 核心功能模块

### 1. IVIM模型 (`utils.py:30-42`)
```python
def ivim_model(f, D, Dstar, bvals):
    """IVIM双指数模型

    S = f * exp(-b * D*) + (1-f) * exp(-b * D)

    参数：
    - f: 灌注分数（perfusion fraction）
    - D: 组织扩散系数（mm²/s）
    - Dstar: 伪扩散系数（mm²/s）
    - bvals: b值序列
    """
```

**IVIM理论要点：**
- **双指数衰减模型**：分离扩散和灌注效应
- **慢速成分**：组织真实扩散 (D)
- **快速成分**：血管内假性扩散 (D*)
- **灌注分数**：血管体积分数 (f)

### 2. 字典生成 (`dictionary_gen.py`)

**功能：**
- 生成IVIM参数字典：覆盖生理参数范围
- SVD基函数提取：降维到2-3个主成分
- 多进程加速计算

**参数范围：**
- f: 0-0.4 (灌注分数)
- D: 0.3e-3 到 3e-3 mm²/s (组织扩散)
- D*: 5e-3 到 60e-3 mm²/s (伪扩散)

**b值设置：**
```python
bvals = [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]
```

### 3. 重建算法 (`utils.py:234-267`)

**核心思想：**
- **局部低秩约束**：利用IVIM信号的低秩特性
- **子空间投影**：基于字典的基函数
- **PICS重建**：使用BART的并行成像压缩感知

**重建流程：**
1. 敏感度图估计
2. 相位校正
3. LLR + 子空间约束重建
4. 参数拟合

### 4. 参数拟合方法

#### 分段拟合 (`utils.py:70-106`)
```python
def ivim_fit_segmented(data, mask, bvals):
    """两步拟合策略：
    1. 高b值(≥400)估计D
    2. 全部b值拟合f和D*

    优点：稳定，快速
    缺点：可能不够精确
    """
```

#### 贝叶斯拟合 (`utils.py:107-137`)
```python
def ivim_fit_bayes(data, mask, nwalkers=75):
    """使用MCMC采样的贝叶斯估计

    优点：提供不确定性量化
    缺点：计算量大
    """
```

## 🚀 使用流程

### 基础使用步骤：

#### 1. 生成字典和基函数
```bash
python dictionary_gen.py --outdir ./output --basis_size 2 --show_figure True
```

#### 2. 处理phantom数据
```bash
python Phantom_IVIM.py
```

#### 3. 重建真实数据
```bash
python IVIM_LLR.py
```

### 高级使用技巧：

#### 1. 调整正则化参数
- `lambda1`: LLR约束强度 (默认0.005)
- `lambda2`: 小波约束强度 (默认0.001)

#### 2. 基函数数量选择
- 通常2-3个基函数足够
- 可通过SVD奇异值分析确定

#### 3. 噪声处理
```python
# 添加复高斯白噪声
noisy_data = add_noise(clean_data, SNR=20)
```

## 📊 算法流程图解析

根据GraphicalAbstract.png，完整算法包含：

**A) 数据采集流程：**
1. 多b值diffusion MRI采集
2. 多线圈接收
3. k空间数据预处理

**B) 信号建模：**
1. IVIM双指数模型
2. 字典生成
3. SVD基函数提取

**C) 重建流程：**
1. 敏感度图估计
2. LLR+子空间约束重建
3. 参数拟合和映射

**D) 结果输出：**
- D图 (组织扩散)
- f图 (灌注分数)
- D*图 (伪扩散)

## 🔍 关键概念详解

### IVIM (Intravoxel Incoherent Motion)
- **物理意义**：体素内水分子的非相干运动
- **组成**：真实扩散 + 微循环灌注
- **应用**：肿瘤、肝脏、肾脏等器官功能评估

### LLR (Locally Low Rank)
- **原理**：相邻体素间参数具有相关性
- **实现**：通过低秩约束减少噪声
- **优势**：保持空间分辨率同时降噪

### 子空间建模 (Subspace Modeling)
- **思想**：用少数基函数表示高维IVIM信号
- **实现**：SVD分解提取主要成分
- **效果**：大幅降低重建复杂度

### 字典学习 (Dictionary Learning)
- **目的**：预先计算参数-信号映射关系
- **方法**：网格搜索 + 并行计算
- **应用**：快速参数估计

## 📈 性能评估指标

### 定量指标
```python
# RMSE计算 (utils.py:381-392)
rmse, ssim_index = calc_rmse(original_image, est_image)

# Bland-Altman分析 (utils.py:324-375)
calc_bland_altman(map_ref, map_calc, name, xlim, plot=True)
```

### 常用评估方法
1. **RMSE/NRMSE**：均方根误差
2. **SSIM**：结构相似性指数
3. **Bland-Altman图**：一致性分析
4. **ROI分析**：感兴趣区域统计

## 🎓 学习路径建议

### 初学者 (1-2周)
1. ✅ **理解IVIM理论**：阅读相关论文和教材
2. ✅ **熟悉项目结构**：了解各模块功能
3. ✅ **运行基础示例**：dictionary_gen.py
4. ✅ **学习BART工具**：MRI重建基础

### 进阶用户 (1-2月)
1. **深入算法原理**：LLR理论和实现
2. **优化参数设置**：针对具体数据调优
3. **扩展功能模块**：添加新的拟合方法
4. **性能分析**：评估不同方法效果

### 高级开发 (3月+)
1. **算法改进**：提出新的约束方法
2. **轨迹扩展**：支持非笛卡尔采样
3. **实时处理**：GPU加速实现
4. **临床验证**：真实数据测试

## 📝 学习笔记

### 重要代码位置记录
- IVIM模型：`utils.py:30-42`
- 字典生成：`dictionary_gen.py:47-105`
- LLR重建：`utils.py:234-267`
- 分段拟合：`utils.py:70-106`
- 贝叶斯拟合：`utils.py:107-137`

### 常见参数设置
```python
# 默认b值
bvals = [0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]

# 参数范围
f_range = [0, 0.4]           # 灌注分数
D_range = [0.3e-3, 3e-3]     # 组织扩散 mm²/s
Dstar_range = [5e-3, 60e-3]  # 伪扩散 mm²/s

# 正则化参数
lambda1 = 0.005  # LLR约束
lambda2 = 0.001  # 小波约束
```

### 待深入学习的内容
- [ ] BART工具链详细用法
- [ ] 敏感度图估计算法
- [ ] 相位校正方法
- [ ] GPU加速实现
- [ ] 临床应用案例

## 📚 扩展阅读

### 相关论文
1. Le Bihan et al. "Separation of diffusion and perfusion in intravoxel incoherent motion MR imaging" (1988)
2. Federau et al. "IVIM perfusion fraction is prognostic for survival in brain glioma" (2017)
3. Zhang et al. "IVIM with locally low rank plus sparse prior" (2022)

### 在线资源
- BART官网：https://mrirecon.github.io/bart/
- DIPY文档：https://dipy.org/
- IVIM教程：https://github.com/oliverchampion/IVIM

---

**联系方式：**
- 项目作者：alan_finkelstein@urmc.rochester.edu
- GitHub：https://github.com/[项目链接]

**笔记更新：**
- 初次整理：2025-11-26
- 下次复习：[待定]
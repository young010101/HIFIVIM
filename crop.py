def brainweb_to_ras(data):
    # data shape: (181, 217, 181)  # (Z:脚→头, Y:后→前, X:左→右)
    data = np.rot90(data, k=1, axes=(0, 2))   # 先绕Y轴转90° → 冠状
    data = np.flip(data, axis=0)              # 上下翻转
    data = np.flip(data, axis=2)              # 左右翻转（可选，根据需要）
    return data

data_ras = brainweb_to_ras(data)

start_y = (217 - 164) // 2   # = 26
start_x = (181 - 164) // 2   # = 8

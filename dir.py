import numpy as np

def dir(mag, dir_angle, top_n=2, suppress_factor=0.4):
    """
    基于梯度方向直方图的梯度抑制
    :param mag: 输入梯度幅值图
    :param dir_angle: 输入梯度方向图（弧度制）
    :param top_n: 保留的主方向数量
    :param suppress_factor: 抑制系数
    :return: 抑制后的梯度幅值图
    """
    # 将弧度转换为角度（0-180度范围）
    dir_deg = np.rad2deg(dir_angle) % 180 # 对于负角度和+180等价
    
    # 构建梯度方向直方图
    hist, bins = np.histogram(dir_deg, bins=18, range=(0, 180))
    
    # 获取主方向bin
    main_bins = np.argsort(hist)[-top_n:]
    main_angles = [(bins[i] + bins[i+1])/2 for i in main_bins]
    
    # 创建方向掩模
    suppress_mask = np.ones_like(dir_deg, dtype=np.float32)
    
    for angle in main_angles:
        # 计算角度差异（考虑周期性）
        angle_diff = np.minimum(np.abs(dir_deg - angle), 
                              180 - np.abs(dir_deg - angle))
        
        # 生成方向抑制掩模
        suppress_mask *= np.where(angle_diff < 10, 1, suppress_factor)
    
    # 应用抑制
    return mag * suppress_mask
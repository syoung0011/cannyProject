import numpy as np
import cv2
import matplotlib.pyplot as plt
import func as fc

def dir(mag, dir_deg, top_n=2, suppress_factor=0.4):
    """
    基于梯度方向直方图的梯度抑制
    :param mag: 输入梯度幅值图
    :param dir_angle: 输入梯度方向图
    :param top_n: 保留的主方向数量
    :param suppress_factor: 抑制系数
    :return: 抑制后的梯度幅值图
    """
    # # 将弧度转换为角度（0-180度范围）
    # dir_deg = dir_angle % 180 # 对于负角度和+180等价
    
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

def scharrDir(img,scharr_scale=1,decay_factor=0.7):
    # 计算x和y方向梯度（Scharr算子，cv2.CV_64F保留负梯度）
    grad_x = cv2.Scharr(src=img, ddepth=cv2.CV_64F, dx=1, dy=0) * scharr_scale
    grad_y = cv2.Scharr(src=img, ddepth=cv2.CV_64F, dx=0, dy=1) * scharr_scale

    # 计算梯度幅值和方向（角度范围0~180°，因边缘方向双向等价）
    grad_mag = np.sqrt(grad_x**2 + grad_y**2) # 梯度幅值
    grad_dir = (np.arctan2(grad_y, grad_x) * 180 / np.pi) % 180 # 梯度方向（0~180°）

    # 统计梯度方向直方图（分箱：每10°为一个区间，共18个区间）
    bin_size = 10 # 每个方向的区间宽度（度）
    num_bins = int(180 / bin_size) # 总区间数（18）
    hist, bins = np.histogram(grad_dir.ravel(), bins=num_bins, range=(0, 180))

    # 找到幅值最大的前2个优势方向区间（文档图像通常水平/垂直主导）
    top2_bins = np.argsort(hist)[-2:] # 索引为0~17，对应0~170°区间（如[8,9]对应80~100°）
    top2_angles = [(bins[i]+bins[i+1])/2 for i in top2_bins] # 优势方向角度范围

    # 构建掩膜：优势方向梯度保留，非优势方向衰减
    mask = np.zeros_like(grad_mag, dtype=np.float32)
    for i in top2_bins:
        start = bins[i]
        end = bins[i+1]
        mask[(grad_dir >= start) & (grad_dir < end)] = 1.0 # 优势方向标记为1

    # 应用衰减：非优势方向梯度幅值 × decay_factor
    adaptive_grad_mag = grad_mag * (mask + (1 - mask) * decay_factor)
    return adaptive_grad_mag,grad_dir,grad_mag

if __name__ == '__main__':
    img=cv2.imread('01.png')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dir1,ang,mag=scharrDir(gray,1,0.1)
    pic=dir(mag,ang,2,0.1)

    plt.figure(figsize=(16,9))
    plt.imshow(dir1,'gray')
    plt.figure(figsize=(16,9))
    plt.imshow(pic,'gray')
    plt.show()
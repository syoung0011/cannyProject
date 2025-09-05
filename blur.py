import cv2
import numpy as np

def gaussBlus(img, sigma_list=[1, 3], weight_list=[0.3, 0.7]):
    """
    多尺度高斯平滑融合
    :param img: 输入灰度图像
    :param sigma_list: 高斯核sigma列表
    :param weight_list: 对应权重列表
    :return: 融合后的平滑图像
    """
    blend = np.zeros_like(img, dtype=np.float32)
    total_weight = sum(weight_list)
    
    for sigma, weight in zip(sigma_list, weight_list):
        # 计算归一化权重
        norm_weight = weight / total_weight
        
        # 生成不同尺度的高斯核
        kernel_size = int(6*sigma + 1)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigmaX=sigma)
        
        # 权重累加
        blend += norm_weight * blurred
    
    # 转换为uint8格式
    return np.clip(blend, 0, 255).astype(np.uint8)

def otherBlur():
    pass
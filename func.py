import cv2
import numpy as np
from blur import gaussBlus,otherBlur
from dir import dir

# ========================
# 1. 基本图像操作
# ========================

# def imRead(name="default.jpg"):
#     img=cv2.imread(name)
#     return img

def imShow(img,name="default",mod=0):
    cv2.imshow(name,img)
    if mod==0:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def imResize(img,max_width=800):
    height,width=img.shape[:2]
    scale=max_width/width
    new_height=(int)(height*scale)
    new_width=(int)(width*scale)
    resize=cv2.resize(img,(new_width,new_height))
    return resize

def imSave(img,name="default.jpg"):
    cv2.imwrite(name,img)

def add_mixed_noise(
    image, 
    gaussian_sigma=20,       # 高斯噪声标准差（控制高斯噪声强度）
    salt_prob=0.05,          # 椒噪声（纯黑）概率（占总噪声比例）
    pepper_prob=0.05,        # 盐噪声（纯白）概率（占总噪声比例）
    is_gray=True             # 输入是否为灰度图（默认文档图像多为灰度）
):
    """
    为图像添加混合噪声（高斯噪声+椒盐噪声），噪声强度可调，效果明显
    参数：
        image: 输入图像（灰度图或彩色图，BGR格式）
        gaussian_sigma: 高斯噪声标准差（越大，高斯噪声越明显）
        salt_prob: 椒噪声（像素值0）占总噪声点的比例
        pepper_prob: 盐噪声（像素值255）占总噪声点的比例
        is_gray: 输入是否为灰度图（影响噪声添加方式）
    返回：
        添加噪声后的图像（uint8格式）
    """
    # 复制原图避免修改原始数据
    noisy_img = np.copy(image).astype(np.float32)  # 转换为浮点型防止溢出
    
    # -------------------------- 高斯噪声 --------------------------
    if gaussian_sigma > 0:
        # 生成与图像尺寸相同的高斯噪声（均值0，标准差sigma）
        if is_gray:
            gaussian_noise = np.random.normal(0, gaussian_sigma, noisy_img.shape)
        else:
            # 彩色图每个通道独立添加高斯噪声
            gaussian_noise = np.random.normal(0, gaussian_sigma, noisy_img.shape)
        noisy_img += gaussian_noise  # 叠加高斯噪声

    # -------------------------- 椒盐噪声 --------------------------
    if salt_prob + pepper_prob > 0:
        # 计算总噪声点数（基于图像像素总数）
        total_pixels = noisy_img.size if is_gray else noisy_img.size // 3  # 彩色图按通道算总像素
        noise_pixels = int(total_pixels * (salt_prob + pepper_prob))
        
        # 随机选择噪声点位置
        if is_gray:
            noise_coords = np.random.choice(noisy_img.size, noise_pixels, replace=False)
            flat_img = noisy_img.flatten()
        else:
            # 彩色图按通道随机选择位置（确保同一位置的三个通道同时被污染）
            noise_coords = np.random.choice(noisy_img.shape[0] * noisy_img.shape[1], 
                                           noise_pixels, replace=False)
            flat_img = noisy_img.reshape(-1, 3)  # 展开为（N,3）的数组
        
        # 生成盐（255）和椒（0）的掩码
        salt_mask = np.random.rand(noise_pixels) < salt_prob
        pepper_mask = ~salt_mask  # 剩余为椒噪声
        
        # 应用椒盐噪声
        if is_gray:
            flat_img[noise_coords[salt_mask]] = 255   # 盐噪声（白色）
            flat_img[noise_coords[pepper_mask]] = 0   # 椒噪声（黑色）
        else:
            flat_img[noise_coords[salt_mask], :] = 255  # 彩色图盐噪声（全白）
            flat_img[noise_coords[pepper_mask], :] = 0  # 彩色图椒噪声（全黑）
        
        # 恢复图像形状
        if not is_gray:
            noisy_img = flat_img.reshape(image.shape)

    # 裁剪像素值到0-255并转换为uint8
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img

# ========================
# 2.传统canny
# ========================

def hysteresis(img, y, x):
    # 递归检查8邻域
    neighbors = [(y-1,x-1), (y-1,x), (y-1,x+1),
                (y,x-1),             (y,x+1),
                (y+1,x-1), (y+1,x), (y+1,x+1)]
    for ny, nx in neighbors:
        if 0 <= ny < img.shape[0] and 0 <= nx < img.shape[1]:
            if img[ny,nx] == 128:  # 弱边缘点
                img[ny,nx] = 255   # 提升为强边缘
                hysteresis(img, ny, nx)  # 递归检查

# ========================
# 3.优化canny
# ========================

# 1.滤波(降噪预处理)
# 高斯滤波优化
# 其他滤波(双边滤波)

# 2.梯度
# 改用Scharr算子
# 方向梯度优化

# 3. 阈值
# otus自适应阈值

# 4.形态学
# 闭运算
# 图形近似

# ========================
# 
# ========================


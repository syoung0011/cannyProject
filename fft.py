import cv2
import numpy as np
import func as fc
# 读取图像并灰度化
img = cv2.imread("11.bmp")
img=fc.imResize(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯平滑去噪（核大小根据噪声程度调整）
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
blurred_float=blurred.astype(np.float32)

# 对预处理后的图像做FFT，并将结果中心化（方便观察）
fft = np.fft.fft2(blurred_float)          # 快速傅里叶变换
fft_shift = np.fft.fftshift(fft)    # 将低频移到中心（原FFT的低频在角落）

rows, cols = blurred.shape
crow, ccol = rows // 2, cols // 2  # 中心坐标
D0 = min(rows, cols) / 10           # 截止频率（根据文档尺寸动态调整）

# 生成高斯低通掩膜（中心亮，周围暗）
mask = np.zeros((rows, cols), np.float32)
for i in range(rows):
    for j in range(cols):
        distance = np.sqrt((i - crow)**2 + (j - ccol)**2)  # 计算当前点到中心的距离
        mask[i, j] = np.exp(-(distance**2)/(2 * (D0**2)))   # 高斯函数（距离越近，值越大）

# 应用低通滤波器（频率域相乘）
filtered_fft_shift = fft_shift * mask  # 关键操作：过滤高频

# 逆FFT（将中心化的频率域转回原始排列）
ifft_shift = np.fft.ifftshift(filtered_fft_shift)
# 逆FFT得到空间域图像（复数转实数）
filtered_img = np.fft.ifft2(ifft_shift)
filtered_img = np.abs(filtered_img)  # 取模（因为逆变换后可能有虚数部分，实际图像是实数）

# ------------------- 新增归一化步骤 -------------------
# 将数值缩放到0-255，并转换为uint8类型（关键！）
filtered_img = cv2.normalize(
filtered_img,
None,
0, 255, # 目标范围0-255
cv2.NORM_MINMAX, # 归一化方法（最小-最大值缩放）
dtype=cv2.CV_8U # 输出类型为uint8
)

fc.imShow(filtered_img)
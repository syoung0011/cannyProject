import cv2
import numpy as np
import func as fc
# ========================
# 1. 图像读取模块
# ========================

#步骤1：读取图片
img_name="11.bmp"
img=cv2.imread(img_name)
fc.imShow(img,"origin")

#步骤2：大小缩放
resize=fc.imResize(img)
fc.imShow(resize,"resize")

# ========================
# 2. Canny算法分步实现
# ========================

# 步骤1：灰度转换（保留原始彩色备用）
gray=cv2.cvtColor(resize,cv2.COLOR_BGR2GRAY)
fc.imShow(gray,"gray",1)

# 步骤2：高斯滤波（降噪）
blur = cv2.GaussianBlur(gray, (5, 5), 0)  # sigma=0让OpenCV自动计算
fc.imShow(blur,"blur")

# 步骤3：Sobel梯度计算
grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
# 计算梯度幅值和方向
magnitude = np.sqrt(grad_x**2 + grad_y**2)
angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
# 可视化梯度
magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
fc.imShow(magnitude_normalized,"sobel")


# 步骤4：非极大值抑制 (NMS)
nms = np.zeros_like(magnitude)
for i in range(1, magnitude.shape[0]-1):
    for j in range(1, magnitude.shape[1]-1):
        direction = angle[i, j]
        # 需要对角度进行标准化,处理负角度的情况
        if direction < 0:
            direction += 180
        # 角度量化到4个方向
        if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
            neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
        elif 22.5 <= direction < 67.5:
            neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
        elif 67.5 <= direction < 112.5:
            neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
        else: # 112.5-157.5
            neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
            
        if magnitude[i, j] >= max(neighbors):
            nms[i, j] = magnitude[i, j]
fc.imShow(nms,"nms")
#magnitude是梯度幅值。本质就是先根据图片尺寸初始化一个全0数组nms（因为默认全部被抑制，然后只要找出极大值
#，给他恢复原来的梯度幅值就行），然后根据这点的角度采用不同的相邻梯度幅值计算公式（对于这四个if，计算两个
#值），只要原来的原梯度幅值都大于这两个值（有一个小都不行，因为就不是极大值）就恢复。

# 步骤5：双阈值处理
high_threshold = 0.2 * np.max(nms)  # 高阈值提高到20%
low_threshold = 0.1 * np.max(nms)   # 低阈值提高到10%
strong_edges = (nms > high_threshold).astype(np.uint8) * 255
weak_edges = ((nms >= low_threshold) & (nms <= high_threshold)).astype(np.uint8) * 255
fc.imShow(strong_edges,"strong",1)
fc.imShow(weak_edges,"weak")

# 步骤6：边缘连接（滞后处理）
edges = np.zeros_like(nms)
edges[nms >= high_threshold] = 255  # 强边缘
edges[(nms >= low_threshold) & (nms < high_threshold)] = 128  # 弱边缘用128标记

# 递归地连接弱边缘
def hysteresis(img, y, x):
    neighbors = [(y-1,x-1), (y-1,x), (y-1,x+1),
                (y,x-1),             (y,x+1),
                (y+1,x-1), (y+1,x), (y+1,x+1)]
    for ny, nx in neighbors:
        if 0 <= ny < img.shape[0] and 0 <= nx < img.shape[1]:
            if img[ny,nx] == 128:
                img[ny,nx] = 255
                hysteresis(img, ny, nx)
height,width=edges.shape[:2]
# 对每个强边缘像素进行连接
for i in range(1, height-1):
    for j in range(1, width-1):
        if edges[i,j] == 255:
            hysteresis(edges, i, j)

# 清除未连接的弱边缘
edges[edges == 128] = 0
fc.imShow(edges,"edges")

# ========================
# 3. Canny算法
# ========================
canny=cv2.Canny(resize,50,150)
fc.imShow(canny,"canny")
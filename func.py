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

# 1.滤波
# 高斯滤波优化
# 其他滤波

# 2.梯度
# 方向梯度优化

# ========================
# 
# ========================


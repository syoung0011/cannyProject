import cv2
import matplotlib.pyplot as plt
import func as fc
import numpy as np
def otsu(img, low_threshold_ratio=0.5):
    # 步骤1：使用Otsu算法计算全局最优阈值
    otsu_threshold, otsu_img = cv2.threshold(
        img,
        0, # 阈值设为0，由Otsu自动计算
        255, # 最大灰度值（二值化后白色为前景）
        cv2.THRESH_BINARY + cv2.THRESH_OTSU # Otsu模式
    )
    print(f"Otsu计算的全局阈值为: {otsu_threshold}")
    # 步骤2：设置Canny双阈值（高阈值为Otsu结果，低阈值为高阈值的1/2）
    high_threshold = np.uint8(otsu_threshold)
    low_threshold = np.uint8(low_threshold_ratio * high_threshold)

    return low_threshold,high_threshold,otsu_img

if __name__ == '__main__':
    img=cv2.imread("image.png")
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    low,high,otus=otsu(gray)

    # plt.figure(figsize=(16,9))
    # plt.imshow(img,cmap='gray')
    # plt.show()
    # plt.figure(figsize=(16,9))
    # plt.imshow(blur,cmap='gray')
    # plt.show()

    print(f"low is {low},high is {high}")
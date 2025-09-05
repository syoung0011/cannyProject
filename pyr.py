import cv2
import numpy as np
import func as fc

def downPyr(img,num=3):
    gaussian_pyramid = [img]
    for _ in range(num - 1):
        # 使用cv2.pyrDown下采样（分辨率减半，先高斯模糊再删行列）
        down_layer = cv2.pyrDown(gaussian_pyramid[-1])
        gaussian_pyramid.append(down_layer)

    # # 输出金字塔各层尺寸（验证下采样正确性）
    # print("高斯金字塔各层尺寸（高分辨率→低分辨率）:")
    # for i, layer in enumerate(gaussian_pyramid):
    #     print(f"层 {i}: {layer.shape[::-1]}") # (宽, 高)
    # #     fc.imShow(layer,"1")
    return gaussian_pyramid

def upDwonPyr(gaussian_pyramid):
    height,width=gaussian_pyramid[0].shape[:2]
    upsampled_edges = []
    for i, edge in enumerate(gaussian_pyramid):
        # 计算需要上采样的次数（当前层是原始的1/(2^i)，需上采样i次恢复原始尺寸）
        up_times = i
        up_edge = edge.copy()

        # 逐次上采样（每次分辨率翻倍）
        for _ in range(up_times):
            up_edge = cv2.pyrUp(up_edge) # pyrUp输出尺寸为输入的2倍

        # 确保上采样后尺寸与原始图像一致（避免因奇数尺寸导致的误差）
        if up_edge.shape != (height,width):
            up_edge = cv2.resize(up_edge, (width,height), interpolation=cv2.INTER_NEAREST)
            

        upsampled_edges.append(up_edge)
    return upsampled_edges
    # for i,layer in enumerate(upsampled_edges):
    #     fc.imShow(layer,"1")

def mergeWeight(upsampled_edges,weights=None):
    height,width=upsampled_edges[0].shape[:2]
    # 初始化融合边缘图（浮点型，避免溢出）
    fused_edge = np.zeros((height,width), dtype=np.float32)
    num = len(upsampled_edges)
    if(weights==None):
        weights = [0.6 - 0.2 * i for i in range(num)]
    # 归一化处理
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights] # 总和变为1

    # 边缘图是0-255的uint8，转为浮点型后加权
    for i,layer in enumerate(upsampled_edges):
        fused_edge += layer.astype(np.float32) * weights[i]

    # 归一化到0-255并转为uint8（边缘图像素值）
    fused_edge = np.clip(fused_edge, 0, 255).astype(np.uint8)
    return fused_edge

if __name__ == "__main__":
    img=cv2.imread("image.png",0)
    fc.imShow(img,"img",1)

    down=downPyr(img)
    up=upDwonPyr(down)
    res=mergeWeight(up)
    fc.imShow(res)

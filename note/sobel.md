#planB（sobel算子原生）
sobel_x=cv2.convertScaleAbs(grad_x)
sobel_y=cv2.convertScaleAbs(grad_y)
sobel_xy=cv2.addWeighted(sobel_x,0.5,sobel_y,0.5,0)
fc.imShow(sobel_xy,"planB")
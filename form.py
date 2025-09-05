import numpy as np
import cv2
import func as fc

img_name="none.jpg"
img=cv2.imread(img_name)
fc.imShow(img,"origin",1)
kernel=np.ones((2,2),np.uint8)
'''
#腐蚀与膨胀
erosion=cv2.erode(img,kernel)
fc.imShow(erosion,"erosion",1)
dilate=cv2.dilate(img,kernel)
fc.imShow(dilate,"dilate")
'''

'''
#开运算与闭运算
open=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
fc.imShow(open,"open",1)

close=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
fc.imShow(close,"close")
'''

'''
#轮廓(膨胀-腐蚀)
gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
fc.imShow(gradient,"gradient")
'''

'''
#礼帽与黑帽
tophat=cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
fc.imShow(tophat,"tophat",1)
blackhat=cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
fc.imShow(blackhat,"blackhat")
'''
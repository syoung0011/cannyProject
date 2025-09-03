import cv2
import numpy as np

def imShow(img,name="none",mod=0):
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

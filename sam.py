from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model



results = model("datasets/test",stream=True)  # predict on an image


for result in results:
    if result.masks is None:
        continue
    maskImg=result.masks.data.cpu().numpy()
    oriImg=result.orig_img
    # 获取原始图像的形状
    ori_shape = oriImg.shape[:2]  # 获取原始图像的高度和宽度

    # 调整 mask 图像大小以匹配原始图像的尺寸
    resized_mask = cv2.resize(maskImg[0], (ori_shape[1], ori_shape[0]), interpolation=cv2.INTER_NEAREST)


    masked_image=oriImg.copy()
    masked_image[resized_mask == 0] = 0  # 将 mask 为 0 的像素对应的原始图像像素置为 0

    xyxy=result.boxes.xyxy.cpu().numpy()[0]
    xyxy=np.round(xyxy).astype(int)
    croped=masked_image[xyxy[1]:xyxy[3],xyxy[0]:xyxy[2],:]

    s=result.path.split("/")
    s[-2]+="-seg"
    os.makedirs("/".join(s[:-1]),exist_ok=True)
    # cv2.imwrite("/".join(s), masked_image)
    cv2.imwrite("/".join(s), croped)
    # break

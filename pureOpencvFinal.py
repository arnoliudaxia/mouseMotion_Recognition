import cv2
import os

import numpy as np
import yaml
from tqdm import tqdm
from collections import deque
from datetime import datetime

# video_path = r"Z:\Motion_Recognition\Head Twitch\largeSlice\output_000.mp4"
# video_path = r"D:\Projects\mouseMotion_Recognition\data\1\out_concat.mp4"
video_path = r'Z:\Motion_Recognition\Head Twitch\largeSlice\output_001.mp4'  # 更换为你的视频路径

# region 参数配置区域
threholdForMMax = 25.0  # 检测阈值
recordWin = 10
# endregion

crop_info = {}
# 从YAML文件中加载裁剪信息
with open('crop_info.yaml', 'r') as yaml_file:
    crop_info = yaml.safe_load(yaml_file)
print(crop_info)

outputFramesStack = deque(maxlen=recordWin)
saveFlag = False
saveCouter = 0
saveName = ""

framesCounter = 1

imgsData = []
# 打开视频文件
cap = cv2.VideoCapture(video_path)
# 获取视频的总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# while True:

_, prev_frame = cap.read()

prev_frame = prev_frame[crop_info['y_start']:crop_info['y_end'], crop_info['x_start']:crop_info['x_end']]
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

for _ in tqdm(range(total_frames)):
    ret, frame = cap.read()
    framesCounter += 1
    if not ret:
        break
    frame = frame[crop_info['y_start']:crop_info['y_end'], crop_info['x_start']:crop_info['x_end']]

    if not ret:
        break

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_gray,  # 上一帧的灰度图像
        next=gray,  # 当前帧的灰度图像
        flow=None,  # 初始化光流矩阵，这里未使用，故为None
        pyr_scale=0.3,  # 金字塔缩放比例，越小意味着构建更多层金字塔，提高检测的精确度和鲁棒性
        levels=3,  # 金字塔层数，表示多尺度（层次）的处理
        winsize=5,  # 平均窗口大小，越大能捕捉到更高速的运动，但也平滑更多
        iterations=3,  # 每层金字塔迭代次数，迭代次数越多，计算越精确但也越慢
        poly_n=5,  # 每个像素邻域的大小，用于多项式展开，一般为5或7
        poly_sigma=1.2,  # 高斯标准差，用于平滑导数，一般与poly_n一起使用
        flags=0  # 操作标志，如设置为cv2.OPTFLOW_USE_INITIAL_FLOW则使用flow输入作为初始流估计
    )
    # 计算光流的幅度和角度
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    nowMaxM = np.max(magnitude)
    nowSumM = np.sum(magnitude)
    # 定义缩放的范围
    new_max = 255
    X_min = 0
    X_max = 20
    magnitude_scaled = ((magnitude - X_min) * new_max) / (X_max - X_min)

    # 生成一个全黑的特征图
    feature_map = np.zeros((magnitude.shape[0], magnitude.shape[1], 3), dtype=np.uint8)
    # 将 magnitude 映射到亮度
    feature_map[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # 将 angle 映射到色相
    angle_in_degrees = ((angle + np.pi) * 180 / np.pi) % 180
    feature_map[..., 0] = angle_in_degrees.astype(np.uint8)

    # 设置饱和度为最大值
    feature_map[..., 1] = 255

    # 将颜色空间转换为BGR到HSV
    feature_map = cv2.cvtColor(feature_map, cv2.COLOR_HSV2BGR)
    outputFramesStack.append([frame, feature_map, str(nowMaxM), str(nowSumM)])

    # ishouang=nowMaxM > threholdForMMax and nowSumM > 50000
    ishouang = nowMaxM > threholdForMMax
    # 更新前一帧的图像和灰度图
    prev_gray = gray
    if saveFlag:
        # 继续保存直到saveCouter
        with open(f'result/{saveName}/logMax.txt', "a+") as f:
            f.writelines(str(nowMaxM) + "\n")
        with open(f'result/{saveName}/logMSum.txt', "a+") as f:
            f.writelines(str(nowSumM) + "\n")

        saveCouter += 1

        output_path = f'result/{saveName}/{recordWin - 1 + saveCouter}.jpg'
        cv2.imwrite(output_path, frame)
        output_path = f'result/{saveName}/mag-color-{recordWin - 1 + saveCouter}.jpg'
        cv2.imwrite(output_path, feature_map)
        # output_path = f'result/{saveName}/mag-color-{10 + saveCouter}.jpg'
        # cv2.imwrite(output_path, feature_map)
        if (saveCouter > recordWin and not ishouang):
            saveFlag = False
            saveCouter = 0
        # if nowMaxM < 10 and saveCouter < 4:
        #     # 假的，不要了删掉
        #     shutil.rmtree(f'result/{saveName}')
        #     saveFlag = False
        #     saveCouter = 0

    elif ishouang:
        # 导出连续帧结果
        saveFlag = True
        saveCouter = 0
        saveName = str(framesCounter) + "-" + str(nowMaxM) + datetime.now().strftime('%Y-%m-%d_%H_%M_%S')

        # 保存！
        os.makedirs(f'result/{saveName}')

        for i, f in enumerate(outputFramesStack):
            output_path = f'result/{saveName}/{i}.jpg'
            cv2.imwrite(output_path, f[0])
            output_path = f'result/{saveName}/mag-color-{i}.jpg'
            cv2.imwrite(output_path, f[1])
            with open(f'result/{saveName}/logMax.txt', "a+") as file:
                file.writelines(f[2] + "\n")
            with open(f'result/{saveName}/logMSum.txt', "a+") as file:
                file.writelines(f[3] + "\n")

cv2.destroyAllWindows()

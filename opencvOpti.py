import cv2
import numpy as np


def visualize_optical_flow_with_crop(video_path,jumpFrames=0,frameByframe=False,plotNum=True):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 跳过开头帧
    for _ in range(jumpFrames):
        ret, _ = cap.read()
        if not ret:
            print("跳过帧时到达视频末尾")
            return

    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        print("无法读取视频")
        return

    # 裁切图像：移除左边200个像素
    prev_frame = prev_frame[:, 200:]

    # 获取裁切后的视频分辨率
    frame_height, frame_width = prev_frame.shape[:2]

    # 转换为灰度图像
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # 创建窗口，自动适应图像大小
    cv2.namedWindow('Optical Flow', cv2.WINDOW_AUTOSIZE)

    # 遍历视频的每一帧
    cv2.waitKey(-1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束

        # 裁切图像：移除左边200个像素
        frame = frame[:, 200:]

        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(
            prev=prev_gray,  # 上一帧的灰度图像
            next=gray,  # 当前帧的灰度图像
            flow=None,  # 初始化光流矩阵，这里未使用，故为None
            pyr_scale=0.5,  # 金字塔缩放比例，越小意味着构建更多层金字塔，提高检测的精确度和鲁棒性
            levels=3,  # 金字塔层数，表示多尺度（层次）的处理
            winsize=15,  # 平均窗口大小，越大能捕捉到更高速的运动，但也平滑更多
            iterations=3,  # 每层金字塔迭代次数，迭代次数越多，计算越精确但也越慢
            poly_n=5,  # 每个像素邻域的大小，用于多项式展开，一般为5或7
            poly_sigma=1.2,  # 高斯标准差，用于平滑导数，一般与poly_n一起使用
            flags=0  # 操作标志，如设置为cv2.OPTFLOW_USE_INITIAL_FLOW则使用flow输入作为初始流估计
        )

        # 选择每隔10个像素
        step = 10
        y, x = np.mgrid[step / 2:frame_height:step, step / 2:frame_width:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T

        # 创建线的起点和终点
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)

        # 绘制线
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, isClosed=False, color=(0, 255, 0))

        # 为每条线绘制点
        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

        ## 画向量场的大小
        # 计算光流的幅度和角度
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        print(np.max(magnitude))

        # 选择每隔40个像素
        if plotNum:
            for y in range(0, frame_height,step):
                for x in range(0, frame_width,step):
                    mag = magnitude[y, x]
                    # 仅绘制幅度大于1的矢量大小，减少图像的拥挤
                    if mag > 1:
                        # text = f"{mag:.1f}"
                        text = f"{int(mag)}"
                        cv2.putText(vis, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                        # print(f"({x, y} {text})")


        # 显示结果
        cv2.imshow('Optical Flow', vis)

        # 按'q'键退出循环
        if cv2.waitKey(-1 if frameByframe else 1) & 0xFF == ord('q'):
            break

        # 更新前一帧的图像和灰度图
        prev_gray = gray

    # 释放视频捕获对象并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = r'Z:\Motion_Recognition\Head Twitch\HT vedio\3mg_m2_h1.avi_20240312_154654.mkv'  # 更换为你的视频路径

    visualize_optical_flow_with_crop(video_path,jumpFrames=0,frameByframe=False,plotNum=False)

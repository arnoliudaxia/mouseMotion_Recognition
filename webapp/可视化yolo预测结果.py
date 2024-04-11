import operator
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from scipy.signal import find_peaks
import plotly.graph_objects as go

with st.chat_message("ai"):
    uploaded_file = st.file_uploader(label="请上传predict video得到的log文件")

if uploaded_file is not None:
    status=st.status("处理中...", expanded=True)
    st.write("注意图表右上角可以expand到全屏")
    # 读取文本文件内容
    file_contents = uploaded_file.read().decode("utf-8")
    lines=file_contents.split("\n")
    lines=[line for line in lines[3:] if "vomit" in line]
    FrameNum=[int(re.findall(r'.*/([^/]+\.png)', line)[0].split(".")[0]) for line in lines]
    classHistory=[[line.split(" ")[-5],float(line.split(" ")[-4][:-1]),i] for i,line in zip(FrameNum,lines)] # 每个元素是[分类，置信度]
    classHistory=sorted(classHistory, key=operator.itemgetter(2))


    blockLen=5000
    if len(classHistory)>blockLen:
        st.info("数据过多，将进行分区处理")
        blockNum=int(len(classHistory)/blockLen)
        blockRange = st.radio(
            "frame区域",
            [f"{i*blockLen}-{(i+1)*blockLen}" for i in range(blockNum)])
        blockRange = [int(blockRange.split("-")[0]), int(blockRange.split("-")[1])]
        classHistory=classHistory[blockRange[0]:blockRange[1]]

    status.write("加载数据到内存")

    df = pd.DataFrame(classHistory, columns=['Class', 'Probability', "Frame"])
    # 设置颜色映射
    color_map = {
        "Normal": "#1E90FF",
        "vomit": "#ef553b",
    }
    # x and y given as array_like objects
    fig = px.scatter(df, x="Frame", y="Probability", color="Class",title="Raw Data",color_discrete_map=color_map)
    # 添加垂直参考线
    # refVmoitStart = [316, 840, 1452, 1881, 2299]
    # for ref in refVmoitStart:
    #     fig.add_vline(x=ref, line_dash='dash', line_color='red', annotation_text='vmoit',
    #                   annotation_position="top left")
    st.plotly_chart(fig)
    status.write("绘制Raw图表")

    classHistoryPureDigital = [[0 if classs[0] == 'Normal' else 1, *classs[1:]] for classs in classHistory]
    data = np.array(classHistoryPureDigital)
    signal_cls = data[:, 0] * 2 - 1
    score = data[:, 1] * signal_cls
    # score = np.exp(data[:, 1]) * signal_cls


    def sliding_window(arr, window_size):
        shape = (arr.size - window_size + 1, window_size)
        strides = (arr.itemsize, arr.itemsize)
        return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


    window_size = st.slider('滑动窗口大小', 2, 20, 5,step=1)
    sliding_window_data = sliding_window(score, window_size)
    sliding_window_score = np.sum(sliding_window_data, axis=1)
    status.write("使用滑动窗口计算分数")



    df = pd.DataFrame(sliding_window_score, columns=['Score'])
    fig = px.line(df, y="Score", title='Sliding Window Score')

    peakHeight = st.slider('FindPeak Height', 0.0, float(window_size), 4.8)
    # 寻找峰值
    peaks, _ = find_peaks(sliding_window_score, height=peakHeight)
    i = np.arange(1, sliding_window_score.shape[0] + 1)
    status.write("寻找score曲线峰值")

    # 绘制信号和峰值
    fig = go.Figure()

    # 添加信号线
    fig.add_trace(go.Scatter(x=i, y=sliding_window_score, mode='lines', name='Signal'))

    # 添加峰值点
    fig.add_trace(go.Scatter(x=i[peaks], y=sliding_window_score[peaks], mode='markers', name='Peaks',
                             marker=dict(symbol='x', size=10,color='#ef553b')))

    # 设置布局
    fig.update_layout(title='滑动窗口峰值')

    refVmoitStart = [316, 840, 1452, 1881, 2280,2773]
    for ref in refVmoitStart:
        fig.add_vline(x=ref, line_dash='dash', line_color='red', annotation_text='vmoit',
                      annotation_position="top left")

    st.plotly_chart(fig)
    status.write("绘制滑动窗口峰值图表")


    status.update(label="全部处理完成！", state="complete")
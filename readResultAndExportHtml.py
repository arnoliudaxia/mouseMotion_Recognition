with open("runs/HTFrameResult.log", "r") as file:
    lines = file.readlines()

lines=lines[3:]
lines=[line for line in lines if "frame" in line]
classHistory=[line.split(" ")[-5] for line in lines]
classHistory=[0 if classs=='Normal' else 1 for classs in classHistory ]
import numpy as np

data=[]

for step,tag in enumerate(classHistory):
    if tag==1:
        data.append(step)

data=np.array(data)

import pandas as pd
# 将数组转换成DataFrame
df = pd.DataFrame(data)

# 将DataFrame保存到excel文件  
df.to_excel('output.xlsx', index=False)


import plotly
import plotly.graph_objects as go
import numpy as np

# 生成一维数据
# data = np.array([1, 2, 5, 6, 7, 100, 110, 120])
# data=data[:5000]
# 创建Plotly的散点图
# fig = go.Figure(data=go.Scatter(x=data, y=np.zeros_like(data), mode='markers'))

# # 设置图表布局和样式
# fig.update_layout(
#     title="Interactive One-dimensional Data Plot",
#     xaxis_title="Time",
#     yaxis_title="",
#     width=800,
#     height=200
# )

# # 显示交互式图表
# fig.show()

# fig.write_html("interactive_plot.html")
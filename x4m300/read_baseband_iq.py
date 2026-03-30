import os
import sys
import binascii
import struct

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation


import os

"""
baseband iq的.dat文件数据格式如下:
Name                  DataType                    Description
FrameCounter;         unsigned integer(32);       A sequential counter from the radar data. Incremented for each data message.;
NumOfBins;            unsigned integer(32);       Number of bins in data set.;
BinLength;            float;                      Length in meters between each bin.;
SamplingFrequency;    float;                      Chip sampling frequency in Hz.;
CarrierFrequency;     float;                      Chip carrier frequency in Hz.;
RangeOffset;          float;                      Start of first range bin in meters.;
i_data;                float array;                Array of NumOfBins float values of the signal i_data.;
q_data;                float array;                Array of NumOfBins float values of the signal q_data.;
"""

def read_all_frames(file_path):
    frames = []
    with open(file_path, "rb") as f: #已二进制的方式打开文件
        while True:
            fixed_part = f.read(24) # 前6项内容，每一项都是32位的，即每一项都占4个字节，所以一共占24个字节， 读取前24个字节的内容。
            if len(fixed_part) < 24:
                break  # End of file or incomplete frame
            
            try: # "=IIffff" 中I 表示是无符号32为整数，f表示单精度浮点数, unpack返回的是一个元组
                FrameCounter, NumOfBins, BinLength, SamplingFrequency, CarrierFrequency, RangeOffset = struct.unpack('=IIffff', fixed_part) #将这24个字节的内容进行解析
                array_size = NumOfBins * 4  # 后两个i_data和q_data 都是长度为NumofBins的数组，数组中每一个值是单精度浮点，故每一个值占4字节，所以一个数组占NumofBins * 4 个字节

                # Read i_data array
                i_data_bytes = f.read(array_size) #读取表示i_data的数组
                if len(i_data_bytes) < array_size:
                    break  # Incomplete frame at end
                i_data = struct.unpack(f'={NumOfBins}f', i_data_bytes) #将该数组进行解析，"=3f"表示3个单精度浮点数
                i_data = np.array(i_data)

                # Read q_data array
                q_data_bytes = f.read(array_size)  #读取表示q_data的数组
                if len(q_data_bytes) < array_size:
                    break  # Incomplete frame at end
                q_data = struct.unpack(f'={NumOfBins}f', q_data_bytes)
                q_data = np.array(q_data)
                Amplitude = np.sqrt(q_data ** 2 + i_data ** 2)

                # Save frame data
                frame = {
                    "FrameCounter": FrameCounter,
                    "NumOfBins": NumOfBins,
                    "BinLength": BinLength,
                    "SamplingFrequency": SamplingFrequency,
                    "CarrierFrequency": CarrierFrequency,
                    "RangeOffset": RangeOffset,
                    "i_data": i_data,
                    "q_data": q_data,
                    "Amplitude": Amplitude,
                }
                frames.append(frame)
            except struct.error:
                break  # Corrupt or incomplete frame
    return frames

# Run the function on the uploaded file
all_frames = read_all_frames("F:/X4M300_radar/data/xethru_recording_20250501_150837_def6216f-da0d-49ca-866f-5ade901d4917/xethru_baseband_iq_20250501_150837.dat")
#将数据保存为csv文件
df = pd.DataFrame(all_frames)
df.to_csv("F:/X4M300_radar/data/xethru_recording_20250501_150837_def6216f-da0d-49ca-866f-5ade901d4917/baseband_output.csv", index=False)
num_of_frames = len(all_frames)
print(all_frames[0])
# 可视化读取的数据
amplitude= [frame["Amplitude"] for frame in all_frames]
num_bins = all_frames[0]["NumOfBins"]
x = list(range(num_bins)) # bin 编号作为 x 轴

# 创建图形对象
fig, ax = plt.subplots()
line, = ax.plot(x, amplitude[0])  # 初始帧

# 设置图形属性
ax.set_xlim(0, num_bins - 1)
ax.set_ylim(min(min(p) for p in amplitude), max(max(p) for p in amplitude))
ax.set_xlabel("Bin Index")
ax.set_ylabel("Amplitude")

# 动画更新函数
def update(frame_idx):
    line.set_ydata(amplitude[frame_idx])
    ax.set_title(f"Radar basebadn iq amplitude")
    return line,

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=len(amplitude), interval=1000 / 17, blit=True)

plt.show()
    

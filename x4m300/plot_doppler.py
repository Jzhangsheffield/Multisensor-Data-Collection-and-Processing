import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import ast
from mpl_toolkits.mplot3d import Axes3D


def get_info(df):
    """用于获取基本绘制信息
    """
    distance = sorted(df["Range"].unique())
    freq_step = df["FrequencyStep"].unique()
    freq_start = df["FrequencyStart"].unique()
    fps = df["FPS"].unique()
    print(f"The frequency start is {freq_start}, the frequency step is {freq_step}, the fps is {fps}, the number of distance is {len(distance)}")
    
    return distance, freq_start, freq_step, fps

def prepare_for_plot(df):
    range_grouped = df.groupby(["FrameCounter", "RangeIdx"])["Doppler Power"].apply( # 到这一步，在没有使用.apply之前，df.groupby(["FrameCounter", "RangeIdx"])["Doppler Power"] 返回的是多个Series, 每一个Series有两个元素，每个元素是一个列表。
        lambda x: [item for sublist in x for item in list(sublist)]  #.apply 则沿着指定的轴，将每一个轴对应的数据做为x逐一处理。
    ).reset_index()
    frame_grouped = range_grouped.groupby("FrameCounter")["Doppler Power"].apply(
        lambda x: np.stack(x.tolist(), axis=0)
    ).reset_index()
    
    shapes = [framecount.shape for framecount in frame_grouped["Doppler Power"]]
    if shapes[-1] != shapes[0]:
        frame_grouped = frame_grouped.iloc[:-1 , :]
        
    doppler_power_matrix_3d = np.abs(np.stack(frame_grouped["Doppler Power"], axis=2))
    doppler_power_matrix_3d = 10 * np.log10(doppler_power_matrix_3d)
        
    return doppler_power_matrix_3d, doppler_power_matrix_3d.shape


# 准备3D绘图动画函数（每个 FrequencyStep 一张图）
def plot_3d_doppler_animation(doppler_3d, freq_axis, distance, freq_step_val):
    fig = plt.figure()
    fig.suptitle("Pulse Doppler Map")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"3D Doppler - FrequencyStep={freq_step_val}")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_zlabel("Power")
    

    X, Y = np.meshgrid(distance, freq_axis)

    # 初始化绘图
    surf = [ax.plot_surface(X.T, Y.T, doppler_3d[:, :, 0], cmap='viridis')]

    def update_3d(frame):
        # ax.collections.clear()  # 清除上一帧
        surf[0].remove()
        surf[0] = ax.plot_surface(X.T, Y.T, doppler_3d[:, :, frame], cmap='viridis')
        return surf[0]

    ani = animation.FuncAnimation(
        fig, update_3d, frames=doppler_3d.shape[2], interval=500, blit=False, repeat=True
    )
    return fig, ani

# 为每个 FrequencyStep 创建 3D 动图
figs_3d = []
anims_3d = []

# 将文件读取进来并查看基本信息
df = pd.read_csv(r"F:/X4M300_radar/data/xethru_recording_20250502_095651_1a8dd80c-dc67-4b7f-bb9b-51bc2893fedf/doppler.csv", skip_blank_lines=True)
df["Doppler Power"] = df["Doppler Power"].apply(ast.literal_eval)

grouped_df = df.groupby("FrequencyStep")
for df_ in grouped_df:
    distance, freq_start, freq_step, fps = get_info(df_[1])
    doppler_power_3d_matrix, doppler_power_3d_matrix_shape = prepare_for_plot(df_[1])
    print(doppler_power_3d_matrix_shape)
    
    freq_axis_1 = np.arange(freq_start[0], 0, freq_step)
    # print(len(freq_axis_1), freq_axis_1.shape)
    freq_axis_2 = np.arange(freq_start[1], abs(freq_start[0]), freq_step)
    # print(len(freq_axis_2), freq_axis_2.shape)
    freq_axis = np.concatenate([freq_axis_1, freq_axis_2])
    print(freq_axis, freq_axis.shape)
    
    assert len(freq_axis) == doppler_power_3d_matrix_shape[1], "freq_axis 的长度，与多普勒3D矩阵不对应"
    
    fig, ani = plot_3d_doppler_animation(doppler_power_3d_matrix, freq_axis, distance, "test")
    # figs_3d.append(fig)
    # anims_3d.append(ani)

plt.show()
    


# # 提取出绘图需要的基本信息
# distance = sorted(df["Range"].unique())
# freq_step = df["FrequencyStep"].unique()
# freq_start = df["FrequencyStart"].unique()
# fps = df["FPS"].unique()
# print(f"The frequency start is {freq_start}, the frequency step is {freq_step}, the fps is {fps}")

# # 提取出所有的偶数行和奇数行，其中偶数行是频率从-8.5-0, 而奇数行频率是0-8.5, 需要将两行拼接起来，构成完整的从-8.5到8.5的值。
# df_even = df.iloc[::2, :]
# df_odd = df.iloc[1::2, :]

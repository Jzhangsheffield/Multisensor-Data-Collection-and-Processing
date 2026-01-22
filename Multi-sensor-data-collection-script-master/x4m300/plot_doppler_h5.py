import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

### 读取h5文件，查看其结构
h5_path = "E:/x4m300/Legacy-SW/data/testdoppler_test_9.h5"
with h5py.File(h5_path, "r") as h5_file:
    dataset_name = list(h5_file.keys())[0]  # Assuming only one dataset
    dataset = h5_file[dataset_name][:]

    # Retrieve all field names (columns)
    field_names = dataset.dtype.names
    print(field_names)

    #将数据转换成DataFrame，并将首个range_idx之前的行删除
    df = pd.DataFrame(dataset)
    # df.to_csv('E:/x4m300/Legacy-SW/data/testdoppler_test_9.csv', index=False)
    first_two_matrix_counter = df['matrix_counter'].unique()[:2]
    last_matrix_counter = df['matrix_counter'].unique()[-2:]
    print(first_two_matrix_counter, last_matrix_counter)
    df_filtered = df[~df["matrix_counter"].isin(first_two_matrix_counter)].reset_index(drop=True)

def get_info(df):
    distance = sorted(df["range"].unique())
    freq_step = df["frequency_step"].unique()
    freq_start = df["frequency_start"].unique()
    fps = df["fps"].unique()
    print(f"The frequency start is {freq_start}, the frequency step is {freq_step}, the fps is {fps}, the number of distance is {len(distance)}")
    return distance, freq_start, freq_step, fps

def prepare_for_plot(df):
    range_grouped = df.groupby(["frame_counter", "range_idx"])["Doppler_power"].apply(
        lambda x: [item for sublist in x for item in list(sublist)]
    ).reset_index()

    frame_grouped = range_grouped.groupby("frame_counter")["Doppler_power"].apply(
        lambda x: np.stack(x.tolist(), axis=0)
    ).reset_index()

    shapes = [framecount.shape for framecount in frame_grouped["Doppler_power"]]
    if shapes[-1] != shapes[0]:
        frame_grouped = frame_grouped.iloc[:-1, :]

    doppler_power_matrix_3d = np.abs(np.stack(frame_grouped["Doppler_power"], axis=2))
    doppler_power_matrix_3d = 10 * np.log10(doppler_power_matrix_3d)

    return doppler_power_matrix_3d, doppler_power_matrix_3d.shape

grouped = list(df_filtered.groupby("frequency_step"))
n_steps = len(grouped)

# 创建 Figure 和子图
fig = plt.figure(figsize=(5 * n_steps, 5))
fig.suptitle("Pulse Doppler 3D Animations")

axes = []
surfaces = []
doppler_all = []
X_all = []
Y_all = []

for i, (step_val, df_) in enumerate(grouped):
    distance, freq_start, freq_step, fps = get_info(df_)
    doppler_3d, shape = prepare_for_plot(df_)

    # 构造频率坐标轴
    freq_axis_1 = np.arange(freq_start[0], 0, freq_step)
    freq_axis_2 = np.arange(freq_start[1], abs(freq_start[0]), freq_step)
    freq_axis = np.concatenate([freq_axis_1, freq_axis_2])
    
    print(freq_axis)

    assert len(freq_axis) == shape[1], "freq_axis 的长度与 Doppler 矩阵不匹配"

    doppler_all.append(doppler_3d)

    ax = fig.add_subplot(1, n_steps, i + 1, projection='3d')
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_zlabel("Power")
    ax.set_ylim(-8.5, 8.5)
    ax.set_yticks(np.round(np.arange(-8.5, 8, 2), 1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    if i == 0:
        ax.set_zlim(-115, 0)
        ax.set_title("Pulse-Doppler Fast")
    else:
        ax.set_title("Pulse-Doppler Slow")
        ax.set_zlim(-85, 0)

    X, Y = np.meshgrid(distance, freq_axis)
    Z = doppler_3d[:, :, 0]
    surf = ax.plot_surface(X.T, Y.T, Z, cmap='viridis')
    
    axes.append(ax)
    surfaces.append([surf])  # 用列表封装便于后续替换
    X_all.append(X)
    Y_all.append(Y)


# 动画更新函数：同时更新所有 subplot 的 surface
def update_all(frame):
    for i in range(n_steps):
        surfaces[i][0].remove()  # 删除旧 surface surfaces[i][0] 是一个 Poly3DCollection 对象, 这里调用的是matplotlib的remove方法。
        Z = doppler_all[i][:, :, frame]
        new_surf = axes[i].plot_surface(X_all[i].T, Y_all[i].T, Z, cmap='viridis')
        surfaces[i][0] = new_surf  # 替换 surface


ani = animation.FuncAnimation(
    fig, update_all, frames=doppler_all[0].shape[2], interval=1000, blit=False, repeat=True
)

plt.tight_layout()
plt.show()





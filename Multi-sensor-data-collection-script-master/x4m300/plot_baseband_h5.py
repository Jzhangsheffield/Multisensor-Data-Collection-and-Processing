import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

### 读取h5文件，查看其结构
h5_path = "E:/x4m300/Legacy-SW/data/testbaseband_test_9.h5"
with h5py.File(h5_path, "r") as h5_file:
    dataset_name = list(h5_file.keys())[0]  # Assuming only one dataset
    dataset = h5_file[dataset_name]

    # Retrieve all field names (columns)
    field_names = dataset.dtype.names
    print(field_names)

    #将数据提取出来
    num_bins = dataset[0]["num_bins"]
    amplitudes = np.array([np.sqrt(row['I_data'] **2 + row['Q_data'] **2) for row in dataset])

    print(num_bins, amplitudes, type(amplitudes))


fig, ax = plt.subplots()
x = np.arange(num_bins)
line, = ax.plot(x, amplitudes[0])
# ax.set_ylim(np.min(I_data_all), np.max(I_data_all))
ax.set_title("Amplitude over Time")
ax.set_xlabel("Bin Index")
ax.set_ylabel("Amplitude")

def update(frame):
    line.set_ydata(amplitudes[frame])
    return line,
                          
ani = animation.FuncAnimation(fig, update, frames=len(amplitudes), interval=50, blit=True)
plt.show()



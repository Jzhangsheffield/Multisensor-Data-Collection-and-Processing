import time
import datetime
import numpy as np
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from pylsl import StreamInfo, StreamOutlet, local_clock               # pylsl ≥1.16 :contentReference[oaicite:1]{index=1}

# ---------------- 1. 连接 MindRove Wi-Fi 板 ----------------
BoardShim.enable_dev_board_logger()
params = MindRoveInputParams()       # 默认 192.168.4.1:4210 / TCP
board_id = BoardIds.MINDROVE_WIFI_BOARD
board = BoardShim(board_id, params)

board.prepare_session()
board.start_stream()                 # 开启内部环形缓冲

# ---------------- 2. 通道索引与采样率 ----------------
emg_ch   = BoardShim.get_emg_channels(board_id)          # 8 ch EMG  :contentReference[oaicite:2]{index=2}
accel_ch = BoardShim.get_accel_channels(board_id)        # 3 ch Accel
gyro_ch  = BoardShim.get_gyro_channels(board_id)         # 3 ch Gyro
ts_ch    = BoardShim.get_timestamp_channel(board_id)     # 1 ch 板载时戳
fs       = BoardShim.get_sampling_rate(board_id)         # 500 Hz     :contentReference[oaicite:3]{index=3}

# ---------------- 3. 创建单一 LSL Outlet ----------------
labels = (
    ["board_ts"] +
    [f"EMG_{i+1}" for i in range(len(emg_ch))] +
    ["Acc_X", "Acc_Y", "Acc_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]
)  # 共 2+8+6 = 16 通道

info = StreamInfo(
    name="MindRove_All",
    type="Mixed",
    channel_count=len(labels),
    nominal_srate=fs,                # 统一按 500 Hz 播出
    channel_format="double64",
    source_id="mindrove_wifi_all",
)

chns = info.desc().append_child("channels")
for lbl in labels:                   # 写入元数据，方便接收端识别 :contentReference[oaicite:4]{index=4}
    ch = chns.append_child("channel")
    ch.append_child_value("label", lbl)
    ch.append_child_value("unit", "raw")
    ch.append_child_value("type", "Mixed")

outlet = StreamOutlet(info)          # 建立数据出口 :contentReference[oaicite:5]{index=5}
print("✅  LSL outlet ready → MindRove_All (16 ch @500 Hz)")

# ---------------- 4. 主循环：推送样本 ----------------
window = 1           # 秒
block  = window * fs # 每次从 SDK 拉 500 列
# block = 1

try:
    while True:
        if board.get_board_data_count() >= block:
            data = board.get_board_data(block)   # shape = (rows, cols)
            # print(data.shape, type(data))

            for col in range(data.shape[1]):
                # sys_time  = datetime.datetime.now().timestamp()                 # 系统时间戳
                board_ts  = data[ts_ch, col]             # 板载时间戳

                sample = np.concatenate(
                    ([board_ts],
                    data[emg_ch,   col],
                    data[accel_ch, col],
                    data[gyro_ch,  col])
                ).astype(np.float64)

                outlet.push_sample(sample.tolist(), local_clock())  # 推荐用 local_clock 做 LSL 时戳 :contentReference[oaicite:6]{index=6}

        else:
            time.sleep(0.002)       # 环形缓冲还没满，稍等
except KeyboardInterrupt:
    print("\n🛑  Ctrl-C → 停止")
finally:
    board.stop_stream()
    board.release_session()
    print("🔌  结束并释放资源")

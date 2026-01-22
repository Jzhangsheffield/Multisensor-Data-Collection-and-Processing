"""
xdf_to_csv.py  ——  把指定 XDF ➜ 导出 CSV
pip install pyxdf pandas
"""
import pyxdf
import pandas as pd
from pathlib import Path

xdf_path   = Path(r"D:\Junxi_data\MULTISENSOR_DATA_COLLECTION\_raw_data_structured\J\mindrove\sub-J_ses-S001_task-DSamplePrepare_run-021_emg.xdf")
csv_path   = xdf_path.with_suffix(".csv")
stream_idx = 0                       # 挑第几条流

# ---------- 1. 读取 XDF ----------
streams, fh = pyxdf.load_xdf(xdf_path)
s = streams[stream_idx]
n_samples, n_chan = s["time_series"].shape
print(f"✔️  载入流 #{stream_idx}: {s['info']['name'][0]} | {n_samples} × {n_chan}")

# ---------- 2. 解析通道标签 ----------
try:
    chan_nodes = (
        s["info"]["desc"][0]         # <desc>
        ["channels"][0]              # <channels>
        ["channel"]                  # <channel> 列表/单节点
    )
    # 如果只有 1 个 <channel>，pyxdf 给的不是 list 而是 dict，手动包成 list
    if not isinstance(chan_nodes, list):
        chan_nodes = [chan_nodes]

    labels = [ch.get("label", [""])[0] or f"Chan_{i+1}"
              for i, ch in enumerate(chan_nodes)]
    # 如果标签数 < n_chan，用占位名补齐
    if len(labels) < n_chan:
        labels += [f"Chan_{i+1}" for i in range(len(labels), n_chan)]
except Exception as e:
    print("⚠️  解析通道标签失败，使用默认名。原因:", e)
    labels = [f"Chan_{i+1}" for i in range(n_chan)]

# ---------- 3. 写 CSV ----------
df = pd.DataFrame(s["time_series"], columns=labels)
df.insert(0, "lsl_timestamp", s["time_stamps"])
df.to_csv(csv_path, index=False, float_format="%.9f")
print("📄 已写入:", csv_path)

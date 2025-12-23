import os
from scipy.io import loadmat
from PIL import Image
import numpy as np

root = r"D:\user\code\AI\My-research\data\pu\pu_DSFSFD\N15_M07_F10"   # 换成你的某个domain文件夹
files = sorted([f for f in os.listdir(root) if f.endswith(".mat")])
mat_path = os.path.join(root, files[0])
stem = files[0].split(".")[0]
jpg_path = os.path.join(root, stem + ".jpg")

m = loadmat(mat_path)
print("keys:", [k for k in m.keys() if not k.startswith("__")])
print("DE_time shape:", m["DE_time"].shape, "dtype:", m["DE_time"].dtype)
print("FFT_data shape:", m["FFT_data"].shape, "dtype:", m["FFT_data"].dtype)

img = Image.open(jpg_path)
print("jpg mode:", img.mode, "size:", img.size)

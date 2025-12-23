from scipy.io import loadmat
m = loadmat(r"D:\user\dataSet\！！工业旋转轴承数据集\德国帕德博恩轴承数据集\K001\N15_M07_F10_K001_1.mat", squeeze_me=True, struct_as_record=False)
print([k for k in m.keys() if not k.startswith("__")])
for k in [k for k in m.keys() if not k.startswith("__")]:
    v = m[k]
    if hasattr(v, "_fieldnames"):
        print("struct var:", k, "fields:", v._fieldnames)

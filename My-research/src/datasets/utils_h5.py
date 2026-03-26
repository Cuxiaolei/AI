import json
import ast

def parse_h5_attr(attr_val):
    """解析HDF5属性"""
    if isinstance(attr_val, bytes):
        attr_val = attr_val.decode("utf-8")
    try:
        return json.loads(attr_val)
    except:
        return ast.literal_eval(attr_val)

def validate_h5_structure(h5_file, require_tf=False):
    """校验HDF5结构（已强制require_tf=False）"""
    required = ["x_freq", "y", "domain"]
    for name in required:
        if name not in h5_file:
            raise ValueError(f"缺失必需数据集: {name}")

    # 长度一致
    lengths = [len(h5_file[k]) for k in required]
    if len(set(lengths)) != 1:
        raise ValueError("x_freq / y / domain 长度不一致")
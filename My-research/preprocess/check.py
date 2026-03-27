# inspect_h5.py
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np


SPLIT_NAMES = {"train", "val", "valid", "validation", "test", "support", "query"}
X_KEYS = {"x", "X", "data", "signals", "signal", "features", "feature", "images", "image"}
Y_KEYS = {"y", "label", "labels", "target", "targets"}
D_KEYS = {"domain", "domains", "d", "domain_id", "domain_ids"}


def decode_if_needed(x: Any) -> Any:
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8")
        except Exception:
            return x
    return x


def to_python_scalar(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    return decode_if_needed(x)


def maybe_parse_json(x: Any) -> Any:
    x = decode_if_needed(x)
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_subheader(title: str) -> None:
    print("\n" + "-" * 100)
    print(title)
    print("-" * 100)


def format_shape(shape: Tuple[int, ...]) -> str:
    return "x".join(str(s) for s in shape)


def summarize_numeric_array(arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    if arr.size == 0:
        return "empty"
    if not np.issubdtype(arr.dtype, np.number):
        return "non-numeric"

    finite = np.isfinite(arr) if np.issubdtype(arr.dtype, np.floating) else np.ones(arr.shape, dtype=bool)
    finite_count = int(finite.sum())
    total_count = int(arr.size)
    nan_count = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    inf_count = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0

    msg = [
        f"min={np.nanmin(arr):.6g}",
        f"max={np.nanmax(arr):.6g}",
        f"mean={np.nanmean(arr):.6g}",
        f"std={np.nanstd(arr):.6g}",
        f"finite={finite_count}/{total_count}",
    ]
    if nan_count > 0:
        msg.append(f"nan={nan_count}")
    if inf_count > 0:
        msg.append(f"inf={inf_count}")
    return ", ".join(msg)


def summarize_small_values(arr: np.ndarray, max_items: int = 10) -> str:
    arr = np.asarray(arr)
    if arr.size == 0:
        return "[]"
    flat = arr.reshape(-1)
    show = flat[:max_items]
    vals = [repr(to_python_scalar(v)) for v in show]
    suffix = " ..." if flat.size > max_items else ""
    return "[" + ", ".join(vals) + "]" + suffix


def print_attrs(obj: h5py.Group, prefix: str = "") -> None:
    if len(obj.attrs) == 0:
        return
    print(prefix + "attrs:")
    for k, v in obj.attrs.items():
        v = maybe_parse_json(v)
        print(prefix + f"  - {k}: {v}")


def walk_h5(name: str, obj: Any, level: int = 0) -> None:
    indent = "  " * level
    short_name = name if name else "/"

    if isinstance(obj, h5py.Group):
        print(f"{indent}[GROUP] {short_name}")
        print_attrs(obj, indent + "  ")
        for key in obj.keys():
            walk_h5(f"{short_name.rstrip('/')}/{key}" if short_name != "/" else f"/{key}", obj[key], level + 1)
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}[DATASET] {short_name} | shape={obj.shape}, dtype={obj.dtype}")
        print_attrs(obj, indent + "  ")


def guess_field(group: h5py.Group, candidates: set) -> Optional[str]:
    for k in group.keys():
        if k in candidates and isinstance(group[k], h5py.Dataset):
            return k
    for k in group.keys():
        if k.lower() in {c.lower() for c in candidates} and isinstance(group[k], h5py.Dataset):
            return k
    return None


def read_dataset(ds: h5py.Dataset) -> np.ndarray:
    return ds[()]


def try_read_mapping(group: h5py.Group, keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in group:
            obj = group[k]
            if isinstance(obj, h5py.Dataset):
                value = obj[()]
                value = maybe_parse_json(value)
                if isinstance(value, np.ndarray):
                    try:
                        return [to_python_scalar(v) for v in value.tolist()]
                    except Exception:
                        return value.tolist()
                return value
    return None


def normalize_1d_int(arr: np.ndarray) -> Optional[np.ndarray]:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        return None
    if arr.dtype.kind in "iu":
        return arr.astype(np.int64)
    if arr.dtype.kind == "f":
        if np.all(np.isfinite(arr)) and np.allclose(arr, np.round(arr)):
            return np.round(arr).astype(np.int64)
    return None


def print_counter(counter: Counter, name: str, mapping: Optional[Any] = None, max_items: int = 100) -> None:
    print(f"{name} distribution (count={sum(counter.values())}):")
    items = sorted(counter.items(), key=lambda x: x[0])
    if len(items) > max_items:
        items = items[:max_items]
        print("  [only first part shown]")
    for k, v in items:
        label_name = None
        if isinstance(mapping, dict):
            label_name = mapping.get(str(k), mapping.get(k, None))
        elif isinstance(mapping, (list, tuple)) and isinstance(k, (int, np.integer)) and 0 <= int(k) < len(mapping):
            label_name = mapping[int(k)]
        extra = f" ({label_name})" if label_name is not None else ""
        print(f"  {k}: {v}{extra}")


def print_joint_distribution(y: np.ndarray, d: np.ndarray, y_map: Optional[Any] = None, d_map: Optional[Any] = None) -> None:
    print("joint distribution of domain x class:")
    table: Dict[int, Counter] = defaultdict(Counter)
    for yy, dd in zip(y.tolist(), d.tolist()):
        table[int(dd)][int(yy)] += 1

    for domain_id in sorted(table.keys()):
        domain_name = None
        if isinstance(d_map, dict):
            domain_name = d_map.get(str(domain_id), d_map.get(domain_id, None))
        elif isinstance(d_map, (list, tuple)) and 0 <= int(domain_id) < len(d_map):
            domain_name = d_map[int(domain_id)]
        domain_title = f"domain {domain_id}" + (f" ({domain_name})" if domain_name is not None else "")
        print(f"  {domain_title}")
        for class_id, cnt in sorted(table[domain_id].items(), key=lambda x: x[0]):
            class_name = None
            if isinstance(y_map, dict):
                class_name = y_map.get(str(class_id), y_map.get(class_id, None))
            elif isinstance(y_map, (list, tuple)) and 0 <= int(class_id) < len(y_map):
                class_name = y_map[int(class_id)]
            extra = f" ({class_name})" if class_name is not None else ""
            print(f"    class {class_id}: {cnt}{extra}")


def inspect_dataset_content(ds_name: str, ds: h5py.Dataset) -> None:
    print_subheader(f"dataset details: {ds_name}")
    arr = read_dataset(ds)
    print(f"shape: {arr.shape}")
    print(f"dtype: {arr.dtype}")
    print(f"sample values: {summarize_small_values(arr)}")

    if arr.size == 0:
        print("status: empty dataset")
        return

    if np.issubdtype(arr.dtype, np.number):
        print(summarize_numeric_array(arr))

    if arr.ndim == 1:
        if arr.dtype.kind in "iuf":
            uniq = np.unique(arr)
            if len(uniq) <= 50:
                print(f"unique values ({len(uniq)}): {uniq.tolist()}")
            else:
                print(f"unique values: {len(uniq)}")
    elif arr.ndim >= 2 and np.issubdtype(arr.dtype, np.number):
        first = np.asarray(arr[0])
        print(f"first sample stats: {summarize_numeric_array(first)}")


def inspect_split(group_name: str, group: h5py.Group) -> None:
    print_header(f"split inspection: {group_name}")
    print(f"keys: {list(group.keys())}")
    print_attrs(group, "  ")

    x_key = guess_field(group, X_KEYS)
    y_key = guess_field(group, Y_KEYS)
    d_key = guess_field(group, D_KEYS)

    y_map = try_read_mapping(group, ["label_map", "y_map", "class_map", "class_names", "labels_text"])
    d_map = try_read_mapping(group, ["domain_map", "domain_names", "domains_text"])

    if x_key is not None:
        x = read_dataset(group[x_key])
        print_subheader(f"main feature field: {x_key}")
        print(f"shape: {x.shape}")
        print(f"dtype: {x.dtype}")
        if x.size > 0 and np.issubdtype(x.dtype, np.number):
            print(summarize_numeric_array(x))
            if x.ndim >= 2:
                print(f"single-sample shape: {x[0].shape}")
                print(f"first sample stats: {summarize_numeric_array(np.asarray(x[0]))}")

    if y_key is not None:
        y = normalize_1d_int(read_dataset(group[y_key]))
        print_subheader(f"label field: {y_key}")
        if y is None:
            print("label field exists, but not a 1D integer-like array")
        else:
            print(f"shape: {y.shape}, num_classes: {len(np.unique(y))}")
            print_counter(Counter(y.tolist()), "label", mapping=y_map)

    if d_key is not None:
        d = normalize_1d_int(read_dataset(group[d_key]))
        print_subheader(f"domain field: {d_key}")
        if d is None:
            print("domain field exists, but not a 1D integer-like array")
        else:
            print(f"shape: {d.shape}, num_domains: {len(np.unique(d))}")
            print_counter(Counter(d.tolist()), "domain", mapping=d_map)

    if x_key is not None and y_key is not None:
        x = read_dataset(group[x_key])
        y_raw = read_dataset(group[y_key])
        y = normalize_1d_int(y_raw)
        if y is not None and x.shape[0] == y.shape[0]:
            print_subheader("consistency check: x vs y")
            print(f"OK: x.shape[0] == y.shape[0] == {x.shape[0]}")
        else:
            print_subheader("consistency check: x vs y")
            print("WARNING: x and y first dimension do not match, or y is not 1D integer-like")

    if x_key is not None and d_key is not None:
        x = read_dataset(group[x_key])
        d_raw = read_dataset(group[d_key])
        d = normalize_1d_int(d_raw)
        if d is not None and x.shape[0] == d.shape[0]:
            print_subheader("consistency check: x vs domain")
            print(f"OK: x.shape[0] == domain.shape[0] == {x.shape[0]}")
        else:
            print_subheader("consistency check: x vs domain")
            print("WARNING: x and domain first dimension do not match, or domain is not 1D integer-like")

    if y_key is not None and d_key is not None:
        y = normalize_1d_int(read_dataset(group[y_key]))
        d = normalize_1d_int(read_dataset(group[d_key]))
        if y is not None and d is not None and len(y) == len(d):
            print_subheader("class-domain joint distribution")
            print_joint_distribution(y, d, y_map=y_map, d_map=d_map)
        else:
            print_subheader("class-domain joint distribution")
            print("WARNING: cannot build domain x class table")

    for k in group.keys():
        if isinstance(group[k], h5py.Dataset) and k not in {x_key, y_key, d_key}:
            inspect_dataset_content(f"{group_name}/{k}", group[k])


def inspect_root_level_datasets(f: h5py.File) -> None:
    has_split_group = False
    for key in f.keys():
        if isinstance(f[key], h5py.Group) and key.lower() in SPLIT_NAMES:
            has_split_group = True
            break

    if has_split_group:
        for key in f.keys():
            if isinstance(f[key], h5py.Group) and key.lower() in SPLIT_NAMES:
                inspect_split(key, f[key])
    else:
        print_header("root-level dataset inspection")
        x_key = guess_field(f, X_KEYS)
        y_key = guess_field(f, Y_KEYS)
        d_key = guess_field(f, D_KEYS)

        pseudo_group = f
        inspect_split("/", pseudo_group)

        if x_key is None and y_key is None and d_key is None:
            print("No standard x/y/domain fields found at root. Full structure above can still be used to inspect manually.")


def main() -> None:

    h5_path = Path(r"D:\user\Documents\ai\paper\1_process\outputs\preprocess\pu\pu_T5_5\train.h5")
    if not h5_path.exists():
        raise FileNotFoundError(f"File not found: {h5_path}")

    print_header("basic file info")
    print(f"path: {h5_path.resolve()}")
    print(f"size: {h5_path.stat().st_size / (1024 ** 2):.2f} MB")

    with h5py.File(h5_path, "r") as f:
        print_header("full h5 structure")
        walk_h5("/", f)

        print_header("root attributes")
        print_attrs(f, "  ")
        if len(f.attrs) == 0:
            print("  no root attrs")

        inspect_root_level_datasets(f)

    print_header("finished")
    print("Inspection completed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_urdf_joints.py — 将 PartNet-Mobility URDF 预处理为 reset 姿态
更新时间：2025-07-13

此次版本要点
-----------
* **彻底移除 Gazebo <initialPosition> 插件** ：很多纯 URDF
  viewer/解析器并不认识 Gazebo 扩展，写进去也看不到效果。
* **STEP 1 恢复为「烘焙 (bake) 零位偏移」**  
  把所有仍为可动关节（revolute / prismatic）的 `limit.lower`
  直接吸收到 `<origin>`，并把 `limit.lower` 归零：  
    • prismatic → 调整 `<origin xyz>`   
    • revolute  → 调整 `<origin rpy>`  
  同时 `limit.upper` 也要减去原来的 lower，使运动范围保持不变。
* 其余流程与此前一致：  
  2. continuous → fixed  
  3. 行程 ≤ 0.15 m 的 prismatic → fixed  
  4. tray 相关 → fixed  
  5. 可动关节排前、固定关节排后并重命名 `joint_0…N`  
  6. 更新 mimic / transmission 引用

统计字段
--------
shifted, continuous_fixed, prismatic_fixed, tray_fixed, renamed
"""

from __future__ import annotations
import sys
from pathlib import Path
import xml.etree.ElementTree as ET


# ---------- 工具 ----------
def _primary_axis(axis_xyz: str):
    ax = [float(v) for v in axis_xyz.strip().split()]
    idx = max(range(3), key=lambda i: abs(ax[i]))
    return idx, (1 if ax[idx] >= 0 else -1)


def _parse_xyz(xyz_str: str | None):
    vals = (xyz_str or "").split()
    vals += ["0"] * (3 - len(vals))
    return [float(v) for v in vals[:3]]


def _parse_rpy(rpy_str: str | None):
    vals = (rpy_str or "").split()
    vals += ["0"] * (3 - len(vals))
    return [float(v) for v in vals[:3]]


def _fmt(vals):
    return " ".join(
        f"{v:.10f}".rstrip("0").rstrip(".") if "." in f"{v:.10f}" else str(int(v))
        for v in vals
    )


# ---------- 单文件处理 ----------
def convert_and_rename_joints(urdf_path: Path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    st = dict(
        shifted=0,
        continuous_fixed=0,
        prismatic_fixed=0,
        tray_fixed=0,
        renamed=0,
    )

    all_joints = root.findall(".//joint")

    # STEP 1 — 烘焙 limit.lower 到 origin ---------------------------------
    for j in all_joints:
        jtype = j.get("type")
        if jtype == "fixed":
            continue
        limit = j.find("limit")
        if limit is None:
            continue
        try:
            lower = float(limit.get("lower", "0"))
            upper = float(limit.get("upper", "0"))
        except ValueError:
            continue
        if abs(lower) < 1e-12:
            continue  # already zero

        # 确保 origin 节点存在
        origin = j.find("origin")
        if origin is None:
            origin = ET.SubElement(j, "origin")
            origin.set("xyz", "0 0 0")

        axis = j.find("axis")
        axis_xyz = axis.get("xyz") if axis is not None else "1 0 0"
        idx, sign = _primary_axis(axis_xyz)

        if jtype == "prismatic":
            xyz = _parse_xyz(origin.get("xyz"))
            xyz[idx] += lower * sign
            origin.set("xyz", _fmt(xyz))
        elif jtype in ("revolute", "continuous"):
            rpy = _parse_rpy(origin.get("rpy"))
            rpy[idx] += lower * sign
            origin.set("rpy", _fmt(rpy))
        else:
            # 其他可动类型（如 planar, floating）此数据集中基本不存在；跳过
            continue

        # 重塑 limit：lower → 0, upper → upper - lower
        limit.set("lower", "0")
        limit.set("upper", f"{upper - lower:.10f}")
        st["shifted"] += 1

    # STEP 2 — continuous → fixed ----------------------------------------
    for j in all_joints:
        if j.get("type") == "continuous":
            j.set("type", "fixed")
            st["continuous_fixed"] += 1

    # STEP 3 — 小行程 prismatic → fixed ----------------------------------
    for j in all_joints:
        if j.get("type") != "prismatic":
            continue
        limit = j.find("limit")
        if limit is None:
            continue
        try:
            lower = float(limit.get("lower", "0"))
            upper = float(limit.get("upper", "0"))
        except ValueError:
            continue
        if upper - lower <= 0.15:
            j.set("type", "fixed")
            st["prismatic_fixed"] += 1

    # STEP 4 — tray → fixed ----------------------------------------------
    tray_links = {
        link.get("name")
        for link in root.findall(".//link")
        if any("tray" in (v.get("name") or "").lower() for v in link.findall("visual"))
    }
    for j in all_joints:
        child = j.find("child")
        if child is not None and child.get("link") in tray_links and j.get("type") != "fixed":
            j.set("type", "fixed")
            st["tray_fixed"] += 1

    # STEP 5 — 排序 & 重命名 ---------------------------------------------
    movable = [j for j in all_joints if j.get("type") != "fixed"]
    fixed = [j for j in all_joints if j.get("type") == "fixed"]
    name_map: dict[str, str] = {}
    for idx, j in enumerate(movable + fixed):
        new_name = f"joint_{idx}"
        old_name = j.get("name")
        if new_name != old_name:
            name_map[old_name] = new_name
            j.set("name", new_name)
            st["renamed"] += 1

    # STEP 6 — 更新 mimic / transmission 引用 ----------------------------
    for m in root.findall(".//mimic"):
        ref = m.get("joint")
        if ref in name_map:
            m.set("joint", name_map[ref])

    for t in root.findall(".//transmission"):
        j_el = t.find("joint")
        if j_el is not None and j_el.get("name") in name_map:
            j_el.set("name", name_map[j_el.get("name")])

    # 保存（若有修改） ----------------------------------------------------
    if any(st.values()):
        tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
    return st


# ---------- 批量入口 ----------
def process_all():
    root_dir = (Path(__file__).resolve().parent.parent / "urdf").resolve()
    if not root_dir.exists():
        print(f"[ERR] 找不到目录: {root_dir}")
        sys.exit(1)

    agg = dict(files=0, mod=0, shifted=0, continuous_fixed=0,
               prismatic_fixed=0, tray_fixed=0, renamed=0)

    for folder in sorted(root_dir.iterdir(), key=lambda p: p.name):
        if not (folder.is_dir() and folder.name.isdigit()):
            continue
        urdf = folder / "mobility.urdf"
        if not urdf.exists():
            continue

        agg["files"] += 1
        st = convert_and_rename_joints(urdf)
        if any(st.values()):
            agg["mod"] += 1
            for k in st:
                agg[k] += st[k]
            print(f"[MOD] {folder.name} | "
                  f"shift {st['shifted']:2}, "
                  f"cont→fx {st['continuous_fixed']:2}, "
                  f"pris→fx {st['prismatic_fixed']:2}, "
                  f"tray→fx {st['tray_fixed']:2}, "
                  f"rename {st['renamed']:2}")

    print("\n===== SUMMARY =====")
    for k, v in agg.items():
        print(f"{k:<18}: {v}")


if __name__ == "__main__":
    process_all()

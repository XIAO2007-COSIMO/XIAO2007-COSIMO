# -*- coding: utf-8 -*-
"""
====================================================================
 车牌颜色阈值识别 · 合成图像原型演示
--------------------------------------------------------------------
 本脚本生成简单绘制的车辆和车牌色块，再用 HSV 阈值统计整张图中的
 绿色/蓝色像素。它是传统图像处理教学原型，不是训练出的 AI 模型，
 也不包含真实车牌定位、字符识别、摄像头接入或硬件联动。
====================================================================
"""
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False


# ============================================================
# 一、生成三张"测试车图"（实际用真实照片替换即可）
# ============================================================
def make_car_image(plate_color):
    """画一辆简单的车，plate_color = 'blue'(燃油) 或 'green'(新能源)。"""
    w, h = 320, 200
    img = Image.new("RGB", (w, h), (210, 215, 220))   # 灰色背景
    d = ImageDraw.Draw(img)
    # 车身
    d.rounded_rectangle([40, 90, 280, 160], radius=18, fill=(120, 130, 140))
    d.rounded_rectangle([80, 60, 240, 100], radius=16, fill=(140, 150, 160))
    # 车轮
    d.ellipse([70, 150, 105, 185], fill=(40, 40, 40))
    d.ellipse([215, 150, 250, 185], fill=(40, 40, 40))
    # 车牌颜色
    if plate_color == "blue":
        plate = (30, 87, 184)      # 蓝底
    else:
        plate = (0, 160, 90)       # 绿底
    d.rectangle([130, 130, 190, 155], fill=plate)
    return np.array(img)


# ============================================================
# 二、核心算法：数绿/蓝高饱和像素
# ============================================================
def detect_vehicle_type(rgb_array):
    """输入一张 RGB 图（numpy），返回判定结果和颜色占比。"""
    arr = rgb_array.astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # --- 转 HSV（matplotlib 的 rgb_to_hsv，H 范围 0~1）---
    hsv = plt.matplotlib.colors.rgb_to_hsv(arr)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # 高饱和、不太暗的像素才是车牌这种"纯色块"
    colorful = (S > 0.4) & (V > 0.3)

    # 绿色：H 大约 0.25 ~ 0.45（对应 90°~160°）
    green = colorful & (H >= 0.25) & (H <= 0.45)
    # 蓝色：H 大约 0.55 ~ 0.68（对应 200°~245°）
    blue = colorful & (H >= 0.55) & (H <= 0.68)

    n_green = int(green.sum())
    n_blue = int(blue.sum())

    # 判定：谁多就是谁
    if n_green > n_blue and n_green > 200:
        verdict = "新能源车（绿牌）"
    elif n_blue > n_green and n_blue > 200:
        verdict = "燃油车（蓝牌）"
    else:
        verdict = "无法判定（车牌不清晰）"

    return verdict, n_green, n_blue


# ============================================================
# 三、跑三张测试图并出结果
# ============================================================
if __name__ == "__main__":
    cases = [
        ("合成样本 A", make_car_image("blue"),  "示意类别：燃油车"),
        ("合成样本 B", make_car_image("green"), "示意类别：新能源车"),
        ("合成样本 C", make_car_image("green"), "示意类别：新能源车"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("车牌颜色阈值识别原型（合成示例）", fontsize=15)

    for ax, (name, img, expect) in zip(axes, cases):
        verdict, n_g, n_b = detect_vehicle_type(img)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{name}\n判定：{verdict}", fontsize=11)
        # 在图下方写颜色统计
        total = max(1, n_g + n_b)
        ax.text(0.5, -0.15,
                f"绿色像素 {n_g}（{n_g/total*100:.0f}%）  "
                f"蓝色像素 {n_b}（{n_b/total*100:.0f}%）\n{expect}",
                transform=ax.transAxes, ha="center", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="#eef6ff", edgecolor="#888"))

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "assets" / "vehicle-type-demo.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[完成] 结果图已保存：{out}")

    # 控制台打印
    print("\n{:<10}{:<20}{:>10}{:>10}".format("样本", "判定结果", "绿色像素", "蓝色像素"))
    print("-" * 55)
    for name, img, expect in cases:
        v, g, b = detect_vehicle_type(img)
        print("{:<10}{:<20}{:>10}{:>10}".format(name, v, g, b))

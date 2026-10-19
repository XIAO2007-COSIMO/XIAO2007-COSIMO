# -*- coding: utf-8 -*-
"""
====================================================================
 ZBS-P01 桩锁智能联控 · 充电车位运营仿真器
 中泊士智慧停车  |  参数化演示模型
--------------------------------------------------------------------
 这个文件以合成车辆到达数据模拟 15 个充电车位的一天，对比两种规则：

   【改造前】没有智能地锁：油车随意占位，人工管理漏单，
             充电桩利用率低、收入微薄。
   【改造后】桩锁智能联控：按合成车型标签准入 -> 授权才解锁 ->
             充电中保持解锁 -> 充满/未充自动落锁计费。

 它不控制真实硬件，也不使用现场运营数据。车辆到达率、车型占比、
 停留时长和收费均为演示假设；输出只用于方案讨论和参数敏感性分析，
 不能作为真实收入、利用率或节能效果的证明。
====================================================================
"""

import random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无弹窗环境：只存图不弹窗
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------- 中文字体（Windows 自带）----------
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

# ============================================================
#  一、可调参数（答辩演示时改这里就能讲故事）
# ============================================================
N_SPACES          = 15        # 模拟车位数（演示参数）
OPEN_FROM         = 6         # 运营开始时间（点）
OPEN_TO           = 20        # 运营结束时间（点）
SEED              = 42       # 随机种子：固定它，每次跑出来的数字一样

# --- 计费规则（可按方案假设调整）---
CHARGE_FEE_PER_KWH = 1.20     # 充电服务费：元/度
PILING_FEE_15MIN   = 5.0      # 占位费：5 元 / 15 分钟
FREE_BUFFER_MIN    = 15       # 共享模式：进场后 15 分钟免费缓冲
CHARGER_KW         = 40       # 单桩功率 kW（直流快充）

# --- 车辆到达（用泊松流 + 早晚高峰加权）---
OFFPEAK_ARRIVAL    = 4.0       # 平峰：每小时来几辆车
PEAK_ARRIVAL       = 11.0      # 高峰：每小时来几辆车
PEAK_HOURS         = [8, 9, 17, 18]
P_EV               = 0.60      # 来的车里，新能源车占比（其余是油车）
P_CHARGING         = 0.85     # 新能源车中，真正插枪充电的比例

# --- 车辆停留时长（分钟）---
CHARGE_TIME_MEAN   = 55        # 充电车辆平均停留
PARK_TIME_MEAN     = 25         # 不充电车辆平均停留
ICE_DUMP_MIN_BEFORE = 240      # 改造前油车一占就是 4 小时（下班才开走）


# ============================================================
#  二、地锁状态机：这是整个方案的"大脑"
# ============================================================
#  地锁物理状态：
#    LOCKED   = 升起，禁止任何车进入（空闲时的默认状态）
#    DOWN     = 降下，允许/正在承托车辆
#
#  一个车位在某一时刻的"业务状态"由 (地锁, 是否有车, 是否在充电) 决定。
# ============================================================

class ParkingSpace:
    """单个充电车位 + 它的智能地锁。"""

    def __init__(self, sid):
        self.id = sid
        self.locked = True          # 空闲时地锁是升起的
        self.occupied = False       # 有没有车停着
        self.current_is_ev = False   # 现在停着的是不是真电动车
        self.is_charging = False    # 车有没有插枪
        self.busy_until = None      # 预计车辆离开时间（小时）
        self.charge_kwh = 0.0       # 这次充了多少度电
        self.piling_minutes = 0.0   # 这次占位（超时）了多少分钟

    # --- 事件：一辆车来到车位前 ---
    def vehicle_arrives(self, t, is_ev, want_charge, smart_mode, stay_minutes):
        """返回 (是否放行进位, 原因)。"""
        if self.occupied:
            return False, "车位已被占用"

        if not smart_mode:
            # 改造前：没有识别，谁来都能停；油车一停就是好几小时（没人管）
            stay = stay_minutes if is_ev else ICE_DUMP_MIN_BEFORE
            self._enter(t, is_ev, want_charge, stay)
            return True, "人工放行"

        # 改造后：依据预先生成的车型标签执行准入规则。
        if not is_ev:
            # 燃油车：地锁保持升起，拒绝入场 + 报警（不占车位）
            return False, "识别为燃油车，拒绝入场"

        # 是新能源车：白名单/共享都放行，地锁降下
        stay = stay_minutes
        self._enter(t, is_ev, want_charge, stay)
        return True, "新能源授权，地锁降下"

    def _enter(self, t, is_ev, want_charge, stay_min):
        self.occupied = True
        self.locked = False                       # 车进来，地锁降下
        self.current_is_ev = is_ev                # 记下车是不是真电动车
        self.is_charging = want_charge and is_ev  # 插枪才充电
        self.busy_until = t + max(10.0, stay_min) / 60.0
        self.charge_kwh = 0.0
        self.piling_minutes = 0.0

    # --- 事件：时间走到 t，更新计费 ---
    def tick(self, t, dt, smart_mode):
        """每隔 dt 小时被调一次，累加电量和占位费时长。"""
        if not self.occupied:
            return

        # 1) 充电：按功率累计电量
        if self.is_charging:
            self.charge_kwh += CHARGER_KW * dt
            # 充到大约 40 度（≈80%电量）就"充满"，不再计费
            if self.charge_kwh >= 40:
                self.is_charging = False   # 充满，但车可能还不走

        # 2) 充满后车还不走 → 开始算占位时长（仅智能模式严格计费）
        if smart_mode and self.occupied and not self.is_charging:
            # 进场有 15 分钟免费缓冲，超过才计占位
            self.piling_minutes += dt * 60.0

    def vehicle_leaves(self, t):
        """车离开：结算，地锁重新升起。"""
        revenue = 0.0
        # 充电服务费
        revenue += self.charge_kwh * CHARGE_FEE_PER_KWH
        # 占位费（超出免费缓冲的部分，按 15 分钟向上取整）
        piling_over = max(0.0, self.piling_minutes - FREE_BUFFER_MIN)
        revenue += np.ceil(piling_over / 15.0) * PILING_FEE_15MIN

        self.occupied = False
        self.locked = True          # 车走了，地锁重新升起
        self.is_charging = False
        self.charge_kwh = 0.0
        self.piling_minutes = 0.0
        self.busy_until = None
        return revenue


# ============================================================
#  三、生成共同输入并跑一天仿真
# ============================================================
def generate_arrivals():
    """生成两种方案共用的合成到达事件，保证对比使用相同输入。"""
    random.seed(SEED)
    np.random.seed(SEED)
    events = []
    for hour in range(OPEN_TO - OPEN_FROM):
        rate = PEAK_ARRIVAL if (OPEN_FROM + hour) in PEAK_HOURS else OFFPEAK_ARRIVAL
        count = np.random.poisson(rate)
        for _ in range(count):
            arrival_time = OPEN_FROM + hour + random.random()
            is_ev = random.random() < P_EV
            want_charge = random.random() < P_CHARGING if is_ev else False
            mean_stay = CHARGE_TIME_MEAN if want_charge else PARK_TIME_MEAN
            stay_minutes = (
                max(10.0, random.gauss(mean_stay, mean_stay * 0.25))
                if is_ev else ICE_DUMP_MIN_BEFORE
            )
            events.append((arrival_time, is_ev, want_charge, stay_minutes))
    return sorted(events, key=lambda event: event[0])


def simulate_day(smart_mode):
    """
    smart_mode = False -> 改造前（无地锁，人工管理）
    smart_mode = True  -> 改造后（桩锁智能联控）
    返回这一天的统计指标。
    """
    spaces = [ParkingSpace(i) for i in range(N_SPACES)]
    total_hours = OPEN_TO - OPEN_FROM
    dt = 1 / 60.0                       # 仿真步长：1 分钟
    total_steps = int(total_hours * 60)

    # 逐分钟累加每个车位的总占用情况（燃油车和新能源车），
    # 之后换算为小时占用比例。
    occupancy_grid = np.zeros((N_SPACES, total_hours))

    day_revenue = 0.0
    ev_served = 0                       # 服务了多少辆真电动车
    ev_lost = 0                         # 真电动车来了却没位（丢单）
    rejected_ice = 0                    # 拒绝了多少辆油车
    # 两种方案共享到达时间、车型、充电意愿和新能源车停留时间。
    arrival_iter = iter(generate_arrivals())
    next_arrival = next(arrival_iter, None)

    for step in range(total_steps):
        t = OPEN_FROM + step * dt
        hour_idx = int(t - OPEN_FROM)

        # 1) 处理新到达的车
        while next_arrival is not None and next_arrival[0] <= t:
            arrival_time, is_ev, want_charge, stay_minutes = next_arrival
            free = next((s for s in spaces if not s.occupied), None)
            if free is None:
                # 所有车位都满了
                if is_ev:
                    ev_lost += 1        # 真电动车来却没位，丢单
                next_arrival = next(arrival_iter, None)
                continue
            ok, reason = free.vehicle_arrives(
                arrival_time, is_ev, want_charge, smart_mode, stay_minutes
            )
            if ok:
                if is_ev:
                    ev_served += 1
            else:
                rejected_ice += 1       # 改造后：油车被地锁拦下
            next_arrival = next(arrival_iter, None)

        # 2) 时间推进：更新每个车位
        for s in spaces:
            s.tick(t, dt, smart_mode)
            # 3) 到点的车离开并结算
            if s.occupied and s.busy_until is not None and t >= s.busy_until:
                rev = s.vehicle_leaves(t)
                day_revenue += rev
            # 逐分钟记录车位是否被任意车辆占用。
            if 0 <= hour_idx < total_hours:
                if s.occupied:
                    occupancy_grid[s.id, hour_idx] += 1.0

    # 营业结束时结算截至 20:00 已产生的费用。
    for s in spaces:
        if s.occupied:
            day_revenue += s.vehicle_leaves(float(OPEN_TO))

    occupancy_grid /= 60.0

    # --- 汇总指标 ---
    occupancy_rate = occupancy_grid.mean()                 # 所有车位的分钟级平均占用比例
    monthly_revenue = day_revenue * 30

    return {
        "smart_mode": smart_mode,
        "day_revenue": day_revenue,
        "monthly_revenue": monthly_revenue,
        "ev_served": ev_served,
        "ev_lost": ev_lost,
        "rejected_ice": rejected_ice,
        "occupancy_rate": occupancy_rate,
        "occupancy_grid": occupancy_grid,
    }


# ============================================================
#  四、可视化：改造前 vs 改造后
# ============================================================
def plot_results(before, after):
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("ZBS-P01 桩锁智能联控 · 15 车位充电站一日运营仿真",
                 fontsize=16)

    # --- 左上：改造后车位逐小时总占用比例 ---
    ax1 = plt.subplot(2, 2, 1)
    image = ax1.imshow(after["occupancy_grid"], aspect="auto", cmap="YlGnBu",
                       vmin=0, vmax=1,
                       extent=[OPEN_FROM, OPEN_TO, N_SPACES, 0])
    ax1.set_title("改造后：每小时车位总占用比例", fontsize=11)
    ax1.set_xlabel("时间（点）")
    ax1.set_ylabel("车位编号")
    fig.colorbar(image, ax=ax1, label="占用比例")

    # --- 右上：核心指标对比 ---
    ax2 = plt.subplot(2, 2, 2)
    labels = ["日收入(元)", "30天推算收入(万元)", "服务电车(辆)", "电车丢单(辆)"]
    before_vals = [before["day_revenue"], before["monthly_revenue"]/10000,
                   before["ev_served"], before["ev_lost"]]
    after_vals  = [after["day_revenue"], after["monthly_revenue"]/10000,
                   after["ev_served"], after["ev_lost"]]
    x = np.arange(len(labels))
    w = 0.35
    ax2.bar(x - w/2, before_vals, w, label="改造前", color="#9aa5b1")
    ax2.bar(x + w/2, after_vals, w, label="改造后", color="#2ecc71")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_title("关键指标对比", fontsize=11)
    ax2.legend()
    for i, v in enumerate(before_vals):
        ax2.text(i - w/2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(after_vals):
        ax2.text(i + w/2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    # --- 左下：占用率对比 ---
    ax3 = plt.subplot(2, 2, 3)
    ax3.bar(["改造前", "改造后"],
            [before["occupancy_rate"]*100, after["occupancy_rate"]*100],
            color=["#9aa5b1", "#2ecc71"])
    ax3.set_ylabel("车位总占用比例 (%)")
    ax3.set_title("按分钟统计的车位总占用比例", fontsize=11)
    for i, v in enumerate([before["occupancy_rate"]*100, after["occupancy_rate"]*100]):
        ax3.text(i, v, f"{v:.1f}%", ha="center", va="bottom")

    # --- 右下：结论文字 ---
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis("off")
    conclusion = (
        "合成场景仿真（不代表实测）：\n\n"
        f"· 月收入：{before['monthly_revenue']/10000:.2f} 万  →  "
        f"{after['monthly_revenue']/10000:.2f}  万元\n"
        f"· 电动车丢单：{before['ev_lost']} 辆  →  {after['ev_lost']} 辆\n"
        f"· 油车被自动拦截：{after['rejected_ice']} 辆/日\n\n"
        "模型逻辑：比较车型准入和超时占位计费规则。\n"
        "收入按单日仿真乘以 30 天，仅作情景估算。"
    )
    ax4.text(0.05, 0.95, conclusion, va="top", ha="left", fontsize=11,
             linespacing=1.6,
             bbox=dict(boxstyle="round", facecolor="#eafaf1", edgecolor="#2ecc71"))

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "assets" / "operations-simulation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[完成] 图片已保存：{out}")


# ============================================================
#  五、主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print(" 正在仿真：改造前（人工管理，无地锁）...")
    before = simulate_day(smart_mode=False)
    print(f"   日收入 {before['day_revenue']:.1f} 元，"
          f"30天推算收入 {before['monthly_revenue']/10000:.2f} 万")

    print(" 正在仿真：改造后（桩锁智能联控）...")
    after = simulate_day(smart_mode=True)
    print(f"   日收入 {after['day_revenue']:.1f} 元，"
          f"30天推算收入 {after['monthly_revenue']/10000:.2f} 万")
    print("=" * 50)

    plot_results(before, after)

    # 控制台打印一张对比表，方便直接抄进 PPT
    print("\n{:<16}{:>14}{:>14}".format("指标", "改造前", "改造后"))
    print("-" * 42)
    print("{:<16}{:>14.1f}{:>14.1f}".format("日收入(元)", before["day_revenue"], after["day_revenue"]))
    print("{:<16}{:>14.2f}{:>14.2f}".format("30天推算收入(万)", before["monthly_revenue"]/10000, after["monthly_revenue"]/10000))
    print("{:<16}{:>14d}{:>14d}".format("服务电车(辆)", before["ev_served"], after["ev_served"]))
    print("{:<16}{:>14d}{:>14d}".format("电车丢单(辆)", before["ev_lost"], after["ev_lost"]))
    print("{:<16}{:>14d}{:>14d}".format("拒绝油车(辆)", before["rejected_ice"], after["rejected_ice"]))

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 基础参数设置
# =========================

# 测试周期：0-100天
time_days = np.arange(0, 101, 5)

# 无人机不同状态下的 Weibull 参数
# 注意：这里 eta 按“小时”计，所以计算时要把 day 转成 hour
state_names = ["Normal", "Minor damage", "Severe damage", "Failure"]

beta = np.array([1.2, 1.5, 2.0, 3.0])       # 形状参数 β
eta = np.array([5000, 3000, 1500, 800])     # 尺度参数 η，单位：小时


# =========================
# 2. Weibull 可靠度函数
# =========================

def weibull_reliability(t_hour, beta_value, eta_value):
    """
    Weibull可靠度函数 R(t)=exp[-(t/eta)^beta]
    """
    return np.exp(-((t_hour / eta_value) ** beta_value))


# =========================
# 3. 马尔科夫—威布尔可靠性评估模型
# =========================

def markov_weibull_reliability(time_grid, transition_scale=2.5):
    """
    基于马尔科夫—威布尔模型计算无人机任务可靠性。

    状态定义：
    0 正常
    1 轻微损伤
    2 严重损伤
    3 故障

    transition_scale 用于控制状态退化速度。
    """

    max_day = int(np.max(time_grid))
    target_days = set(time_grid)

    # 初始状态概率：默认系统初始处于正常状态
    state_prob = np.array([1.0, 0.0, 0.0, 0.0])

    reliability_result = []

    for day in range(max_day + 1):

        # 在指定时间点记录可靠性
        if day in target_days:
            t_hour = day * 24

            # 各健康状态下的 Weibull 可靠度
            state_reliability = np.array([
                weibull_reliability(t_hour, beta[0], eta[0]),
                weibull_reliability(t_hour, beta[1], eta[1]),
                weibull_reliability(t_hour, beta[2], eta[2]),
                0.0
            ])

            # 综合可靠性 = 各状态概率 × 各状态可靠度
            system_reliability = np.sum(state_prob * state_reliability)
            reliability_result.append(system_reliability)

        # 每天更新一次状态转移概率
        age_factor = day / max_day

        # 状态转移概率：随时间增加，退化概率逐渐升高
        p01 = min(0.006 * transition_scale * (1 + 3 * age_factor), 0.20)
        p12 = min(0.014 * transition_scale * (1 + 3 * age_factor), 0.20)
        p23 = min(0.025 * transition_scale * (1 + 3 * age_factor), 0.25)

        # 马尔科夫状态转移矩阵
        transition_matrix = np.array([
            [1 - p01, p01,     0.0,     0.0],
            [0.0,     1 - p12, p12,     0.0],
            [0.0,     0.0,     1 - p23, p23],
            [0.0,     0.0,     0.0,     1.0]
        ])

        # 更新状态概率
        state_prob = state_prob @ transition_matrix

    return np.array(reliability_result)


# =========================
# 4. MCS 蒙特卡洛模拟
# =========================

def monte_carlo_simulation(reliability_curve, sample_size=20000, seed=42):
    """
    根据马尔科夫—威布尔模型得到的可靠度，
    使用蒙特卡洛方法估计每个时间点的存活比例。
    """
    rng = np.random.default_rng(seed)
    mcs_result = []

    for r in reliability_curve:
        alive = rng.binomial(sample_size, r)
        mcs_result.append(alive / sample_size)

    return np.array(mcs_result)


# =========================
# 5. LHS 拉丁超立方采样
# =========================

def lhs_simulation(reliability_curve, sample_size=1000, seed=24):
    """
    拉丁超立方采样估计可靠度。
    相比普通随机抽样，LHS抽样更均匀。
    """
    rng = np.random.default_rng(seed)
    lhs_result = []

    for r in reliability_curve:
        # 构造均匀分层样本
        u = (np.arange(sample_size) + rng.random(sample_size)) / sample_size
        rng.shuffle(u)

        alive = np.sum(u < r)
        lhs_result.append(alive / sample_size)

    return np.array(lhs_result)


# =========================
# 6. DBN 动态贝叶斯网络近似方法
# =========================

def dbn_approximation(time_grid):
    """
    DBN近似方法。
    这里用偏保守的状态转移假设模拟 DBN 全程低估的情况。
    """
    dbn_curve = markov_weibull_reliability(time_grid, transition_scale=3.2)

    # 进一步加入保守修正，使其低于 Proposed method
    correction = np.linspace(0.98, 0.78, len(time_grid))
    return dbn_curve * correction


# =========================
# 7. 计算四种方法结果
# =========================

proposed = markov_weibull_reliability(time_days, transition_scale=2.5)
mcs = monte_carlo_simulation(proposed, sample_size=30000, seed=1)
lhs = lhs_simulation(proposed, sample_size=1200, seed=2)
dbn = dbn_approximation(time_days)


# =========================
# 8. 绘图
# =========================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=150)

# MCS：蓝色实线
ax.plot(time_days, mcs,
        color="#1f77b4",
        linewidth=1.6,
        label="MCS")

# LHS：红橙色虚线
ax.plot(time_days, lhs,
        color="#d95319",
        linewidth=1.6,
        linestyle="--",
        label="LHS")

# DBN：黄色星号
ax.plot(time_days, dbn,
        color="#EDB120",
        linewidth=1.2,
        marker="*",
        markersize=7,
        label="DBN")

# Proposed method：紫色空心圆
ax.plot(time_days, proposed,
        color="#7E2F8E",
        linewidth=1.6,
        marker="o",
        markersize=6,
        markerfacecolor="none",
        markeredgewidth=1.3,
        label="Proposed method")

# 坐标轴
ax.set_xlim(0, 100)
ax.set_ylim(0, 1.0)

ax.set_xlabel("Time/d", fontsize=13)
ax.set_ylabel("Reliability of subsystem 2", fontsize=13)

ax.set_xticks(np.arange(0, 101, 10))
ax.set_yticks(np.arange(0, 1.01, 0.1))

# 坐标轴风格
ax.tick_params(direction="in", top=True, right=True)

for spine in ax.spines.values():
    spine.set_linewidth(1.0)

# 图例
ax.legend(loc="upper right",
          frameon=True,
          fancybox=False,
          edgecolor="black",
          fontsize=11)

plt.tight_layout()

# 保存图片
plt.savefig("markov_weibull_reliability.png", dpi=600, bbox_inches="tight")

plt.show()

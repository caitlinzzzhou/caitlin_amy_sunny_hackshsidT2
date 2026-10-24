"""
This code is the functions from main.py that are implemented in datacenter_gui.py.
"""

import pulp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# plot colors and style (indigo + emerald, muted warm accents)
NAVY, BLUE, TEAL, GOLD, RED, GRAY = "#3B5BA5", "#A9C7E8", "#2CA58D", "#F2B134", "#D64550", "#9AA5B1"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.6,
})

# ---------- parameters and tasks ----------
def default_params():
    return dict(
        H=24, dt=1.0,
        C=10000.0, eta=0.67,
        B_max=3000.0, B_init=1000.0, P_batt_max=1500.0,
        eta_ch=0.95, eta_dis=0.95,
        pi_g=[0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4,
              0.4, 0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.5,
              0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
        pi_t=[0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9,
              1.0, 1.2, 1.3, 1.3, 1.2, 1.1, 1.0, 1.0,
              1.1, 1.2, 1.3, 1.2, 1.1, 1.0, 0.8, 0.6],
        G=[0.0, 0.0, 0.0, 0.0, 2000.0, 3000.0, 5000.0, 6000.0,
           7000.0, 8000.0, 8000.0, 9000.0, 9000.0, 8000.0,
           7000.0, 6000.0, 5000.0, 4000.0, 3000.0, 2000.0,
           1000.0, 0.0, 0.0, 0.0],
        gamma_g=0.02, gamma_t=0.394,
        alpha=1.0, beta=0.283,
        green_scale=1.0,    # scales green supply (green scarcity test)
        high_scale=1.0,     # scales HIGH urgency work (urgency test)
    )


# (start hour, end hour, number of tasks, kWh per task)
HIGH_PERIODS = [(8, 12, 114, 80), (12, 14, 54, 80), (14, 18, 152, 80), (18, 22, 50, 80)]
MED_PERIODS  = [(0, 6, 40, 50), (6, 8, 55, 50), (8, 12, 72, 50), (12, 14, 95, 50),
                (14, 18, 80, 50), (18, 22, 50, 50), (22, 24, 20, 50)]
LOW_PERIODS  = [(0, 6, 60, 30), (6, 8, 70, 30), (18, 22, 40, 30), (22, 24, 15, 30)]


def make_tasks(P):
    # same tasks as Main MILP.py, just with an urgency label added
    # (greedy needs it) and the HIGH work scaled by high_scale
    tasks, tid = [], 1

    def add(arrival, deadline, work, urg):
        nonlocal tid
        tasks.append({"id": tid, "a": arrival, "d": min(deadline, P["H"] - 1),
                      "w": work, "u": urg})
        tid += 1

    for s, e, cnt, uw in HIGH_PERIODS:
        wph = (cnt * uw) / (e - s) * P["high_scale"]
        for h in range(s, e):
            add(h, h + 1, wph, "HIGH")
    for s, e, cnt, uw in MED_PERIODS:
        wph = (cnt * uw) / (e - s)
        for h in range(s, e):
            add(h, h + 4, wph, "MED")
    for s, e, cnt, uw in LOW_PERIODS:
        wph = (cnt * uw) / (e - s)
        for h in range(s, e):
            add(h, 23, wph, "LOW")
    return tasks


# ---------- MILP ----------
def solve_milp(P, tasks):
    H, dt = P["H"], P["dt"]
    b_max, b_init = P["B_max"], min(P["B_init"], P["B_max"])

    model = pulp.LpProblem("Data_Center_Power_Computing_Co_Scheduling", pulp.LpMinimize)

    # x[j, t] = work of task j done in hour t (only inside its window)
    x = {}
    for task in tasks:
        j = task["id"]
        for t in range(task["a"], task["d"] + 1):
            x[(j, t)] = model.add_variable(f"x_{j}_{t}", lowBound=0.0, cat="Continuous")
    P_g   = [model.add_variable(f"P_g_{t}", lowBound=0.0) for t in range(H)]
    P_t   = [model.add_variable(f"P_t_{t}", lowBound=0.0) for t in range(H)]
    P_ch  = [model.add_variable(f"P_ch_{t}", lowBound=0.0) for t in range(H)]
    P_dis = [model.add_variable(f"P_dis_{t}", lowBound=0.0) for t in range(H)]
    B     = [model.add_variable(f"B_{t}", lowBound=0.0, upBound=b_max) for t in range(H + 1)]
    z     = [model.add_variable(f"z_{t}", cat="Binary") for t in range(H)]

    # every task has to be fully done
    for task in tasks:
        j = task["id"]
        model += (pulp.lpSum(x[(j, t)] for t in range(task["a"], task["d"] + 1)) == task["w"],
                  f"Task_Completion_{j}")

    # capacity, power balance, green limit
    for t in range(H):
        active = [x[(task["id"], t)] for task in tasks if task["a"] <= t <= task["d"]]
        load = pulp.lpSum(active) / P["eta"]
        model += (load <= P["C"], f"Capacity_{t}")
        model += (load + P_ch[t] == P_g[t] + P_t[t] + P_dis[t], f"Power_Balance_{t}")
        model += (P_g[t] <= P["G"][t] * P["green_scale"], f"Green_Limit_{t}")

    # battery
    model += (B[0] == b_init, "Battery_Init")
    # has to end the day at least as full as it started, otherwise the
    # solver just drains the starting charge for free
    model += (B[H] >= b_init, "Battery_End_SoC")
    for t in range(H):
        model += (B[t + 1] == B[t] + P["eta_ch"] * P_ch[t] * dt - (P_dis[t] / P["eta_dis"]) * dt,
                  f"SoC_{t}")
        model += (P_ch[t] <= P["P_batt_max"] * z[t], f"ChBound_{t}")
        model += (P_dis[t] <= P["P_batt_max"] * (1 - z[t]), f"DisBound_{t}")

    Cost = pulp.lpSum((P["pi_g"][t] * P_g[t] + P["pi_t"][t] * P_t[t]) * dt for t in range(H))
    Carbon = pulp.lpSum((P["gamma_g"] * P_g[t] + P["gamma_t"] * P_t[t]) * dt for t in range(H))
    model += P["alpha"] * Cost + P["beta"] * Carbon, "Total_Objective"

    solver = get_solver()
    stats = model.solve(solver)

    if stats.status != pulp.LpSolveStatus.Optimal:
        return None

    sched = {(j, t): pulp.value(v) for (j, t), v in x.items() if pulp.value(v) > 1e-6}
    return dict(
        status=stats.status.name,
        sched=sched,
        P_g=[pulp.value(v) or 0.0 for v in P_g],
        P_t=[pulp.value(v) or 0.0 for v in P_t],
        P_ch=[pulp.value(v) or 0.0 for v in P_ch],
        P_dis=[pulp.value(v) or 0.0 for v in P_dis],
        B=[pulp.value(v) or 0.0 for v in B],
        cost=pulp.value(Cost), carbon=pulp.value(Carbon),
    )


def get_solver():
    # CBC if it's installed, HiGHS otherwise
    try:
        s = pulp.COIN_CMD(msg=False)
        if s.available():
            return s
    except Exception:
        pass
    return pulp.HiGHS(msg=False)


# ---------- greedy baseline ----------
def greedy_schedule(P, tasks):
    # no real scheduling: each hour, run whatever has arrived (most urgent
    # first) until capacity is full. No battery, never waits for a cheaper hour.
    H = P["H"]
    cap = P["C"] * P["eta"]                      # compute we can deliver per hour
    remaining = {task["id"]: task["w"] for task in tasks}
    sched = {}
    P_g, P_t = [0.0] * H, [0.0] * H
    rank = {"HIGH": 0, "MED": 1, "LOW": 2}

    for t in range(H):
        waiting = [k for k in tasks if k["a"] <= t <= k["d"] and remaining[k["id"]] > 1e-9]
        waiting.sort(key=lambda k: (rank[k["u"]], k["d"], k["a"]))
        used = 0.0
        for k in waiting:
            if used >= cap - 1e-9:
                break
            do = min(remaining[k["id"]], cap - used)
            if do > 1e-9:
                sched[(k["id"], t)] = do
                remaining[k["id"]] -= do
                used += do
        demand = used / P["eta"]                 # electricity needed this hour
        if P["pi_g"][t] < P["pi_t"][t]:          # use green first if it's cheaper
            g = min(P["G"][t] * P["green_scale"], demand)
            P_g[t], P_t[t] = g, demand - g
        else:
            P_t[t] = demand
    return sched, P_g, P_t


# ---------- metrics ----------
def compute_metrics(P, tasks, sched, P_g, P_t):
    dt = P["dt"]
    cost = sum((P["pi_g"][t] * P_g[t] + P["pi_t"][t] * P_t[t]) * dt for t in range(P["H"]))
    carbon = sum((P["gamma_g"] * P_g[t] + P["gamma_t"] * P_t[t]) * dt for t in range(P["H"]))
    ok = 0
    for task in tasks:
        done = sum(sched.get((task["id"], t), 0.0) for t in range(task["a"], task["d"] + 1))
        if done >= task["w"] - 1e-4:
            ok += 1
    sla = 100.0 * ok / len(tasks)
    green_total = sum(P["G"]) * P["green_scale"]
    green_util = 100.0 * sum(P_g) / green_total if green_total > 0 else 0.0
    peak = max(P_t)
    return dict(cost=cost, carbon=carbon, sla=sla, green_util=green_util, peak=peak)


# ---------- figures ----------

def fig_schedule(res, P):
    H = P["H"]
    hours = np.arange(H)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    ax.bar(hours, res["P_g"], 0.8, color=TEAL, edgecolor="white", linewidth=0.4, label="Green")
    ax.bar(hours, res["P_t"], 0.8, bottom=res["P_g"], color=NAVY, edgecolor="white",
           linewidth=0.4, label="Grid")
    ax.bar(hours, res["P_dis"], 0.8, bottom=np.array(res["P_g"]) + np.array(res["P_t"]),
           color=GOLD, edgecolor="#B27A16", linewidth=0.4, label="Battery discharge")
    load = [sum(v for (j, tt), v in res["sched"].items() if tt == t) / P["eta"] for t in range(H)]
    chg = np.array(res["P_ch"]) / P["eta_ch"]
    ax.plot(hours, np.array(load) + chg, color=RED, lw=2.0, ls="--",
            marker="o", ms=4, label="Demand (compute + charging)")
    ax.set_ylabel("Power (kWh per hour)", labelpad=8)
    ax.margins(y=0.10)
    ax.set_title("24h schedule from the MILP", pad=12)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=9,
              handlelength=1.4, columnspacing=1.2, borderaxespad=0.2)

    ax2.fill_between(hours, res["B"][1:], color=BLUE, alpha=0.40, step="mid")
    ax2.plot(hours, res["B"][1:], color=NAVY, lw=1.5, drawstyle="steps-mid")
    ax2.set_ylabel("Battery (kWh)", labelpad=8)
    ax2.set_xlabel("Hour of day", labelpad=6)
    ax2.set_xlim(-0.5, H - 0.5)
    return fig

def fig_greedy_vs_milp(m_milp, m_greedy):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    dcost = 100 * (m_greedy["cost"] - m_milp["cost"]) / m_greedy["cost"]
    dcarb = 100 * (m_greedy["carbon"] - m_milp["carbon"]) / m_greedy["carbon"]
    for ax, (key, title, unit, delta) in zip(axes, [
            ("cost", "Cost", "¥", dcost),
            ("carbon", "Carbon", "kg CO$_2$", dcarb)]):
        vals = [m_milp[key], m_greedy[key]]
        bars = ax.bar(["MILP", "Greedy"], vals, 0.55, color=[NAVY, GRAY])
        ax.set_title(f"{title} ({unit})")
        ax.bar_label(bars, fmt="%.0f", padding=2)
        ax.text(0.5, 0.92, f"MILP is {delta:.1f}% lower", transform=ax.transAxes,
                ha="center", fontsize=11, color=RED, fontweight="bold")
    return fig

def fig_gantt(res, tasks, P):
    """The actual schedule: one row per task (pale = allowed window, solid = ran),
    power sources aligned underneath on the same time axis."""
    order = sorted(tasks, key=lambda k: ({"HIGH": 0, "MED": 1, "LOW": 2}[k["u"]], k["a"], k["id"]))
    n = len(order)
    colors = {"HIGH": "#D64550", "MED": "#F2B134", "LOW": "#2CA58D"}
    prefix = {"HIGH": "H", "MED": "M", "LOW": "L"}
    counters, labels = {}, []
    for k in order:
        counters[k["u"]] = counters.get(k["u"], 0) + 1
        labels.append(f"{prefix[k['u']]}{counters[k['u']]}")

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11, 0.20 * n + 3.4), sharex=True,
        gridspec_kw={"height_ratios": [n, 3]}, constrained_layout=True)

    # window (pale) + executed (solid) per task
    for i, k in enumerate(order):
        y = n - 1 - i
        ax.barh(y, k["d"] - k["a"] + 1, left=k["a"] - 0.5, height=0.78,
                color="#EDF0F4", edgecolor="#C9CED6", linewidth=0.5, zorder=1)
        for t in range(k["a"], k["d"] + 1):
            if res["sched"].get((k["id"], t), 0.0) > 1e-6:
                ax.barh(y, 0.92, left=t - 0.46, height=0.62,
                        color=colors[k["u"]], zorder=2)

    # group separators + group labels on the right edge
    n_high = sum(1 for k in order if k["u"] == "HIGH")
    n_med = sum(1 for k in order if k["u"] == "MED")
    for y in (n - n_high - 0.5, n - n_high - n_med - 0.5):
        ax.axhline(y, color="#B8BEC7", lw=0.8, ls=":")
    ax.text(23.6, n - n_high / 2, "HIGH", fontsize=9, color=colors["HIGH"],
            fontweight="bold", va="center")
    ax.text(23.6, n - n_high - n_med / 2, "MED", fontsize=9, color="#B8860B",
            fontweight="bold", va="center")
    ax.text(23.6, (n - n_high - n_med) / 2, "LOW", fontsize=9, color=colors["LOW"],
            fontweight="bold", va="center")

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1], fontsize=6)
    ax.set_ylabel("Task", labelpad=6)
    ax.set_xlim(-0.6, 23.6)
    ax.set_title("Schedule showing when each task runs", pad=10)
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=colors[u], label=f"{u} task") for u in ["HIGH", "MED", "LOW"]]
    handles.append(mpatches.Patch(color="#EDF0F4", edgecolor="#C9CED6", label="Allowed window"))
    ax.legend(handles=handles, loc="upper left", frameon=False, ncol=4, fontsize=8,
              handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)

    # aligned power stack, charging shown below zero
    hours = np.arange(P["H"])
    ax2.bar(hours, res["P_g"], 0.8, color=TEAL, label="Green")
    ax2.bar(hours, res["P_t"], 0.8, bottom=res["P_g"], color=NAVY, label="Grid")
    bottom = np.array(res["P_g"]) + np.array(res["P_t"])
    ax2.bar(hours, res["P_dis"], 0.8, bottom=bottom, color=GOLD,
            edgecolor="#B27A16", linewidth=0.4, label="Battery discharge")
    ax2.bar(hours, -np.array(res["P_ch"]) / P["eta_ch"], 0.8, color=GOLD, alpha=0.40,
            edgecolor="#B27A16", linewidth=0.4, label="Battery charge")
    ax2.axhline(0, color="#8A8F98", lw=0.8)
    ax2.margins(y=0.12)
    ax2.set_ylabel("Power (kWh/h)", labelpad=6)
    ax2.set_xlabel("Hour of day", labelpad=6)
    ax2.legend(loc="upper left", frameon=False, ncol=3, fontsize=8,
               handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)
    return fig

#----------tables------------

def table_tasks(res, tasks):
    rows = []
    for task in tasks:
        j = task["id"]
        hours_run = [t for t in range(task["a"], task["d"] + 1)
                    if res["sched"].get((j, t), 0.0) > 1e-6]
        total = sum(res["sched"].get((j, t), 0.0) for t in range(task["a"], task["d"] + 1))
        rows.append({
            "id": j,
            "urgency": task["u"],
            "window": f"{task['a']}–{task['d']}",
            "power (kW)": round(task["w"], 1),
            "hours run": hours_run,
        })

    df_tasks = pd.DataFrame(rows)
    return df_tasks

def table_hourly_power(P, res):
    df_hourly_power = pd.DataFrame({
        "hour": range(P["H"]),
        "green (kWh)": res["P_g"],
        "grid (kWh)": res["P_t"],
        "charge (kWh)": res["P_ch"],
        "discharge (kWh)": res["P_dis"],
        "battery state of charge (kWh)": res["B"][1:],
    })
    return df_hourly_power
    

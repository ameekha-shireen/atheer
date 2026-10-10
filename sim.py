"""
ATHEER — Predictive Network Handover Simulation

Compares:
  1. baseline         — reactive handover (wait for failure)
  2. predictive_hard  — ML prediction, hard switch
  3. predictive_mbb   — ML prediction + make-before-make
  4. satellite_only   — always-on satellite fallback

Features:
  - 4 network types (wired, wifi, 5G, satellite)
  - 14 5G towers with overlapping coverage
  - 4 apps with different priorities
  - 3 user speeds (pedestrian, car, train)
  - 50 simulated trips per scenario (independent trips, no shared congestion)
  - Random mid-trip network failures
  - Regret metric (gap from oracle)
  - Per-handover event log
  - Handover effectiveness summary

Run:  python3 sim.py
"""
import os, csv, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------- 1. WORLD
NETS = ["wired", "wifi", "5G", "satellite"]
COST = np.array([0.0, 0.0, 1.0, 3.0])
L = 20.0
CELL = 0.005
NCELL = int(L / CELL) + 1
GRID = np.arange(NCELL) * CELL

WIFI_SPOTS = [0.05, 0.6, 3.0, 15.0]
WIFI_R = 0.15

# 14 5G towers with overlapping coverage (only real gap: ~9.0-9.5 km)
TOWERS = [0.5, 1.5, 2.5, 4.0, 5.5, 6.5, 7.5, 11.0, 12.5, 14.0, 15.0, 17.0, 18.0, 19.5]
TOWER_R = 1.5

CANYON = (3.5, 5.0)
WIRED_END = 0.03

SAT_TP, SAT_LAT, SAT_LOSS = 55.0, 45.0, 1.0
EPOCH = 15
DIP_TP, DIP_LAT, DIP_LOSS = 0.3, 2.0, 3.0
RHO = 0.97


def mean_map():
    tp = np.zeros((4, NCELL)); lat = np.full((4, NCELL), 1000.0); loss = np.full((4, NCELL), 100.0)
    m = GRID < WIRED_END
    tp[0, m], lat[0, m], loss[0, m] = 900, 2, 0.0

    d = np.min(np.abs(GRID[:, None] - np.array(WIFI_SPOTS)[None, :]), axis=1)
    s = np.clip(1 - d / WIFI_R, 0, 1); m = s > 0
    tp[1, m] = 150 * s[m] ** 0.7; lat[1, m] = 10 + 40 * (1 - s[m]); loss[1, m] = 0.2 + 6 * (1 - s[m]) ** 2

    d = np.min(np.abs(GRID[:, None] - np.array(TOWERS)[None, :]), axis=1)
    s = np.clip(1 - d / TOWER_R, 0, 1); m = s > 0
    tp[2, m] = 300 * s[m] ** 0.8; lat[2, m] = 18 + 40 * (1 - s[m]); loss[2, m] = 0.1 + 4 * (1 - s[m]) ** 2

    m = ~((GRID >= CANYON[0]) & (GRID <= CANYON[1]))
    tp[3, m], lat[3, m], loss[3, m] = SAT_TP, SAT_LAT, SAT_LOSS
    return tp, lat, loss


TP0, LAT0, LOSS0 = mean_map()


def ar1(rng, n, rho=RHO):
    z = np.zeros(n); e = rng.standard_normal(n); z[0] = e[0]
    k = np.sqrt(1 - rho ** 2)
    for i in range(1, n):
        z[i] = rho * z[i - 1] + k * e[i]
    return z


def make_trip(rng, start_frac=0.0):
    tp = TP0.copy(); lat = LAT0.copy(); loss = LOSS0.copy()
    sig = [0.0, 0.35, 0.35, 0.15]
    daily = [1.0, 1.0, rng.uniform(0.6, 1.0), rng.uniform(0.6, 1.0)]
    for n in range(1, 4):
        fade = np.exp(sig[n] * ar1(rng, NCELL) - sig[n] ** 2 / 2) * daily[n]
        avail = TP0[n] > 0
        tp[n] = np.where(avail, TP0[n] * fade, 0.0)
        lat[n] = np.where(avail, LAT0[n] * (1 + np.clip(1 - fade, 0, 1)), 1000.0)

    # Random mid-trip network failure
    drift_point = rng.integers(NCELL // 4, 3 * NCELL // 4)
    drift_type = rng.choice(["5G_overload", "wifi_offline", "satellite_block"])
    if drift_type == "5G_overload":
        tp[2, drift_point:] *= 0.4
        lat[2, drift_point:] *= 1.8
    elif drift_type == "wifi_offline":
        tp[1, drift_point:] = 0.0
        lat[1, drift_point:] = 1000.0
    elif drift_type == "satellite_block":
        tp[3, drift_point:] *= 0.3
    return tp, lat, loss


# ----------------------------------------------------------------- 2. APPS / QoE
APPS = {
    "emergency/bank": ("critical", 1.0),
    "voice call": ("call", 0.9),
    "video meeting": ("meeting", 0.8),
    "file download": ("download", 0.2),
}


def lat_score(lat, good, bad):
    return np.clip((bad - lat) / (bad - good), 0, 1)


def qoe(kind, tp, lat, loss):
    ls = np.clip(1 - loss / 8, 0, 1)
    if kind == "call":
        return (tp >= 0.1) * lat_score(lat, 100, 400) * ls
    if kind == "meeting":
        return np.clip(tp / 4, 0, 1) * lat_score(lat, 150, 500) * ls
    if kind == "download":
        return np.clip(tp / 50, 0, 1) * np.clip(1 - loss / 15, 0, 1)
    if kind == "critical":
        return (tp >= 0.2) * lat_score(lat, 300, 1500) * np.clip(1 - loss / 10, 0, 1)
    raise ValueError(kind)


# ----------------------------------------------------------------- 3. PREDICTOR
class Predictor:
    def __init__(self, n_train_trips=60, seed=1):
        rng = np.random.default_rng(seed)
        X = GRID[::2].reshape(-1, 1)
        Y = [[] for _ in range(4)]
        for _ in range(n_train_trips):
            tp, lat, loss = make_trip(rng)
            for n in range(4):
                Y[n].append(np.stack([tp[n, ::2], lat[n, ::2], loss[n, ::2]], 1))
        self.pred = np.zeros((4, NCELL, 3))
        self.r2 = {}
        for n in range(4):
            y = np.concatenate(Y[n]); x = np.tile(X, (n_train_trips, 1))
            mu, sd = y.mean(0), y.std(0) + 1e-9
            rf = RandomForestRegressor(n_estimators=40, min_samples_leaf=40, n_jobs=-1, random_state=0)
            rf.fit(x, (y - mu) / sd)
            self.pred[n] = rf.predict(GRID.reshape(-1, 1)) * sd + mu
        rng2 = np.random.default_rng(999)
        for n in range(4):
            res = tot = 0.0
            for _ in range(10):
                tp, _, _ = make_trip(rng2)
                res += ((tp[n] - self.pred[n, :, 0]) ** 2).sum()
                tot += ((tp[n] - tp[n].mean()) ** 2).sum()
            self.r2[NETS[n]] = 1 - res / tot


# ----------------------------------------------------------------- 4. POLICIES
THR = 2.0
DETECT = 3
HARD = 1.5
MBB = 0.2
HYST = 0.05
COSTW = 0.25
PREF = [0, 1, 2, 3]

HANDOVER_LOG = []


def run_baseline(tp, q, user_id, app):
    T = tp.shape[1]; usable = (tp >= THR).tolist(); qa = q.tolist()
    cur = next((n for n in PREF if usable[n][0]), 3)
    bad = 0; rem = 0.0; sw = 0; intr = 0.0
    sel = np.zeros(T, int); out = np.zeros(T)
    for t in range(T):
        better = next((n for n in PREF if usable[n][t]), None)
        new = None
        if better is not None and better < cur:
            new = better; bad = 0
        elif not usable[cur][t]:
            bad += 1
            if bad >= DETECT and better is not None and better != cur:
                new = better; bad = 0
        else:
            bad = 0
        if new is not None:
            q_before = float(qa[cur][t])
            q_after = float(qa[new][t])
            HANDOVER_LOG.append({
                "user": user_id, "app": app, "policy": "baseline", "time": t,
                "from": NETS[cur], "to": NETS[new],
                "qoe_before": round(q_before, 4), "qoe_after": round(q_after, 4),
                "delta": round(q_after - q_before, 4),
                "beneficial": int(q_after > q_before),
            })
            cur = new; rem += HARD; sw += 1; intr += HARD
        lost = min(1.0, rem); rem -= lost
        sel[t] = cur; out[t] = qa[cur][t] * (1 - lost)
    return sel, out, sw, intr


def run_predictive(score, q, interr, user_id, app, policy_name):
    T = q.shape[1]; sc = score.T.tolist(); qa = q.tolist()
    cur = int(np.argmax(score[:, 0])); rem = 0.0; sw = 0; intr = 0.0
    sel = np.zeros(T, int); out = np.zeros(T)
    for t in range(T):
        s = sc[t]; best = max(range(4), key=lambda n: s[n])
        if best != cur and s[best] > s[cur] + HYST:
            q_before = float(qa[cur][t])
            q_after = float(qa[best][t])
            HANDOVER_LOG.append({
                "user": user_id, "app": app, "policy": policy_name, "time": t,
                "from": NETS[cur], "to": NETS[best],
                "qoe_before": round(q_before, 4), "qoe_after": round(q_after, 4),
                "delta": round(q_after - q_before, 4),
                "beneficial": int(q_after > q_before),
            })
            cur = best; rem += interr; sw += 1; intr += interr
        lost = min(1.0, rem); rem -= lost
        sel[t] = cur; out[t] = qa[cur][t] * (1 - lost)
    return sel, out, sw, intr


def metrics(q):
    out = q < 0.2
    drops = int(np.sum(out[1:] & ~out[:-1]) + (1 if out[0] else 0))
    return q.mean(), int(out.sum()), drops


# ----------------------------------------------------------------- 5. EXPERIMENT
USERS = {
    "pedestrian (5 km/h)": (5.0, 4.0),
    "car (90 km/h)": (90.0, 20.0),
    "train (160 km/h)": (160.0, 20.0),
}
POLICIES = ["baseline", "predictive_hard", "predictive_mbb", "satellite_only"]

N_USERS = 50


def starlink_dynamics(rng, atp, alat, aloss, T):
    ep = np.arange(T) // EPOCH; ne = int(ep.max()) + 1
    f_tp = np.exp(0.45 * rng.standard_normal(ne) - 0.45 ** 2 / 2)[ep]
    f_lat = np.exp(0.25 * rng.standard_normal(ne) - 0.25 ** 2 / 2)[ep]
    f_loss = np.exp(0.6 * rng.standard_normal(ne) - 0.6 ** 2 / 2)[ep]
    b = (np.arange(T) % EPOCH == 0)
    f_tp = np.where(b, f_tp * DIP_TP, f_tp); f_lat = np.where(b, f_lat * DIP_LAT, f_lat)
    add_loss = np.where(b, DIP_LOSS, 0.0)
    m = atp[3] > 0
    atp[3] = np.where(m, np.clip(atp[3] * f_tp, 1.0, 200.0), 0.0)
    alat[3] = np.where(m, alat[3] * f_lat, 1000.0)
    aloss[3] = np.where(m, aloss[3] * f_loss + add_loss, 100.0)


def simulate_trip(rng, predictor, speed, dist, user_id="U", start_offset_km=0.0):
    tp, lat, loss = make_trip(rng)
    v = speed / 3600.0
    # FIX: stop the trip at the end of the route instead of letting the user
    # sit still at the last point (which distorted car/train results).
    dist = min(dist, L - 0.01 - start_offset_km)
    T = int(dist / v)
    x = np.arange(T) * v + start_offset_km
    x = np.minimum(x, L - 0.01)
    cell = np.clip((x / CELL).astype(int), 0, NCELL - 1)
    H = max(10.0, 0.1 / v)
    fracs = [0.25, 0.5, 0.75, 1.0]
    pcells = [np.clip(((x + v * H * f) / CELL).astype(int), 0, NCELL - 1) for f in fracs]
    atp, alat, aloss = tp[:, cell], lat[:, cell], loss[:, cell]
    starlink_dynamics(rng, atp, alat, aloss, T)
    res = {}
    for app, (kind, prio) in APPS.items():
        qa = np.stack([qoe(kind, atp[n], alat[n], aloss[n]) for n in range(4)])
        acc = qa.copy()
        for pc in pcells:
            p = predictor.pred[:, pc, :]
            acc += np.stack([qoe(kind, p[n, :, 0], p[n, :, 1], p[n, :, 2]) for n in range(4)])
        score = acc / (1 + len(pcells)) - COSTW * (1 - prio) * COST[:, None]
        res[app] = {
            "baseline": run_baseline(atp, qa, user_id, app),
            "predictive_hard": run_predictive(score, qa, HARD, user_id, app, "predictive_hard"),
            "predictive_mbb": run_predictive(score, qa, MBB, user_id, app, "predictive_mbb"),
            "satellite_only": (np.full(T, 3), qa[3], 0, 0.0),
        }
        oracle_q = np.max(qa, axis=0)
        for p in POLICIES:
            sel, q, sw, intr = res[app][p]
            regret = float(np.mean(oracle_q - q))
            res[app][p] = (sel, q, sw, intr, regret)
    return res, x


def main(n_trips=N_USERS):
    HANDOVER_LOG.clear()
    print("Training predictor on 60 past trips ...")
    pred = Predictor()
    print("Hold-out R2 of predicted throughput per network:",
          {k: round(float(v), 2) for k, v in pred.r2.items()})
    print(f"\nRunning {n_trips} simulated trips per scenario ...\n")

    rows = []; keep = None
    for u, (speed, dist) in USERS.items():
        agg = {(a, p): [] for a in APPS for p in POLICIES}
        for i in range(n_trips):
            rng = np.random.default_rng(1000 + i)
            start_km = (i / n_trips) * (L * 0.5)
            user_id = f"{u.split()[0]}-{i:03d}"
            res, x = simulate_trip(rng, pred, speed, dist, user_id, start_km)
            if u.startswith("car") and i == 0:
                keep = res
            for a in APPS:
                for p in POLICIES:
                    sel, q, sw, intr, regret = res[a][p]
                    m, o, d = metrics(q)
                    agg[(a, p)].append((m, o, d, sw, intr, regret))
        for (a, p), v in agg.items():
            v = np.array(v).mean(0)
            rows.append(dict(user=u, app=a, policy=p, mean_qoe=v[0], outage_s=v[1],
                             drop_events=v[2], switches=v[3], interruption_s=v[4],
                             regret=v[5]))
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()})
    with open(os.path.join(OUT, "handover_events.csv"), "w", newline="") as f:
        fieldnames = ["user", "app", "policy", "time", "from", "to",
                      "qoe_before", "qoe_after", "delta", "beneficial"]
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for event in HANDOVER_LOG:
            w.writerow(event)
    print(f"Handover events logged: {len(HANDOVER_LOG)}")

    # --- Handover effectiveness summary ---
    stats = collections.defaultdict(lambda: {"total": 0, "beneficial": 0})
    for e in HANDOVER_LOG:
        key = (e["policy"], e["app"])
        stats[key]["total"] += 1
        stats[key]["beneficial"] += e["beneficial"]

    print("\nHANDOVER EFFECTIVENESS SUMMARY")
    print("-" * 70)
    print(f"{'policy':<18}{'app':<18}{'total':>8}{'beneficial':>12}{'rate':>8}")
    print("-" * 70)
    for (pol, app), s in sorted(stats.items()):
        pct = 100 * s["beneficial"] / max(1, s["total"])
        print(f"{pol:<18}{app:<18}{s['total']:>8}{s['beneficial']:>12}{pct:>7.1f}%")
    print("-" * 70)

    with open(os.path.join(OUT, "handover_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "app", "total", "beneficial", "beneficial_rate"])
        for (pol, app), s in sorted(stats.items()):
            pct = 100 * s["beneficial"] / max(1, s["total"])
            w.writerow([pol, app, s["total"], s["beneficial"], round(pct, 2)])

    plots(rows, keep, pred)
    summary(rows)


# ----------------------------------------------------------------- 6. OUTPUT
COL = {"satellite_only": "#d1495b", "baseline": "#8a8f98", "predictive_hard": "#e0a030", "predictive_mbb": "#1f6fd1"}
LAB = {"satellite_only": "Satellite always on", "baseline": "Baseline (reactive)",
       "predictive_hard": "Predictive (hard switch)", "predictive_mbb": "Predictive + MBB"}


def plots(rows, keep, pred):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for n, c in zip(range(4), ["#444", "#2a9d8f", "#1f6fd1", "#d1495b"]):
        ax.plot(GRID, np.maximum(TP0[n], 0.5), color=c, label=NETS[n], lw=1.6)
        ax.plot(GRID, np.maximum(pred.pred[n, :, 0], 0.5), color=c, lw=0.8, ls="--")
    ax.set_yscale("log"); ax.set_xlabel("Position along route (km)"); ax.set_ylabel("Throughput (Mbps, log)")
    ax.axvspan(*CANYON, color="#d1495b", alpha=0.08); ax.text(CANYON[0] + .05, 700, "city: satellite blocked", fontsize=8)
    # FIX: label now matches the real 5G gap (towers at 7.5 and 11.0 km, radius 1.5 km -> gap 9.0-9.5 km)
    ax.axvspan(9.0, 9.5, color="#1f6fd1", alpha=0.15); ax.text(9.6, 700, "5G gap", fontsize=8)
    ax.set_title("Network map (solid = truth, dashed = what the AI learned)"); ax.legend(ncol=4, fontsize=8, loc="lower right")
    ax.set_ylim(0.4, 1500); fig.tight_layout(); fig.savefig(os.path.join(OUT, "1_network_map.png"), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    apps = list(APPS)
    for ax, u in zip(axes, USERS):
        for j, p in enumerate(POLICIES):
            vals = [next(r["mean_qoe"] for r in rows if r["user"] == u and r["app"] == a and r["policy"] == p) for a in apps]
            ax.bar(np.arange(len(apps)) + (j - 1.5) * 0.2, vals, 0.2, color=COL[p], label=LAB[p])
        ax.set_xticks(range(len(apps))); ax.set_xticklabels([a.replace("/", "/\n").replace(" ", "\n", 1) if " " in a else a for a in apps], fontsize=8)
        ax.set_title(u); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Mean experience score (0-1)"); axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "2_experience_score.png"), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, u in zip(axes, USERS):
        for j, p in enumerate(POLICIES):
            vals = [next(r["outage_s"] for r in rows if r["user"] == u and r["app"] == a and r["policy"] == p) for a in apps]
            ax.bar(np.arange(len(apps)) + (j - 1.5) * 0.2, vals, 0.2, color=COL[p])
        ax.set_xticks(range(len(apps))); ax.set_xticklabels([a.replace("/", "/\n").replace(" ", "\n", 1) if " " in a else a for a in apps], fontsize=8)
        ax.set_title(u)
    axes[0].set_ylabel("Seconds unusable per trip")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "3_outage_seconds.png"), dpi=150); plt.close(fig)

    a = "video meeting"; fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    T = len(keep[a]["baseline"][1]); km = np.arange(T) * 90 / 3600
    for p in ["baseline", "predictive_mbb"]:
        sel = keep[a][p][0]; axes[0].step(km, sel + (0.12 if p == "predictive_mbb" else 0), where="post", color=COL[p], label=LAB[p], lw=1.6)
        k = 15; q = np.convolve(keep[a][p][1], np.ones(k) / k, mode="same")
        axes[1].plot(km, q, color=COL[p], label=LAB[p], lw=1.6)
    axes[0].set_yticks(range(4)); axes[0].set_yticklabels(NETS); axes[0].set_title("Car trip, video meeting: which network is used")
    axes[1].set_ylabel("Experience (15 s avg)"); axes[1].set_xlabel("Position along route (km)"); axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8, loc="lower left"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "4_timeline_car.png"), dpi=150); plt.close(fig)


def summary(rows):
    print("\nMean experience score (0-1) | outage seconds per trip")
    for u in USERS:
        print(f"\n{u}")
        for a in APPS:
            line = f"  {a:15s}"
            for p in POLICIES:
                r = next(r for r in rows if r["user"] == u and r["app"] == a and r["policy"] == p)
                line += f" | {p[:14]:14s} {r['mean_qoe']:.2f} / {r['outage_s']:5.0f}s"
            print(line)


if __name__ == "__main__":
    main()
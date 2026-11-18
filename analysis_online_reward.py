# -*- coding: utf-8 -*-
"""Mechanism check for the online-learning harm (reviewer round 1, item B3).

Question: does the online Thompson-ridge scorer raise its OWN average per-kick reward
r = -(C_after_LS - C_before) / median_edge_length relative to the same linear scorer kept static
(linprior)?  If yes while the final gap gets worse, the reward is misaligned with final quality
("myopic reward"); if no, a bug is suspected.  We also log the fraction of "neutral" kicks
(local search returns exactly to the pre-kick cost, r = 0) and the immediate (pre-LS) kick damage.

online_targeting.py / fast_ils.py are hash-registered and NOT modified.  The kick kernel below
(run_kicks_probe) reuses their compiled primitives and is a statement-for-statement copy of
online_targeting.run_kicks_online with extra *accounting only* (no extra RNG draws, identical
updates of tour / A / b / stats); test_online_reward.py checks bit-identity against the original.
run_probe() mirrors online_targeting.run_online, replacing only the kernel call.

Development instances only (vlsi_primary, 12 spread over n), 3 seeds, 10 s, single process:
    py -3.13 analysis_online_reward.py run
    py -3.13 analysis_online_reward.py hl          # Hodges-Lehmann for V1-V4 on holdout2_10s (no runs)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

import fast_ils as F
from fast_ils import EPS, local_search
import online_targeting as OT
from online_targeting import D, _kick_global_cuts, _ref_row

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "kick_targeting"
NEUTRAL_TOL = 1e-6          # |r| below this (in median-edge units) counts as a neutral kick
WINDOWS = ((0.0, 1.0), (1.0, 5.0), (5.0, 10.0))

# accumulator layout (float64)
A_KICKS, A_SUMR, A_IMPR, A_NEUT, A_SUMDMG, A_ROWOK, A_SUMR_ROWOK, A_SUMACC, A_RESTORED, \
    A_SUMR2, A_NEUT_ROWOK, A_WORSE = range(12)
NACC = 12


@njit(cache=True)
def run_kicks_probe(tour, pos, btour, bpos, cost, n, xs, ys, kind, nbr, n_kicks,
                    guided, pool_u, pool_v, pool_m, prob, queue, inq, qht, buf, stats,
                    X, ref_tour, ref_pos, inv_scale, A, bvec, learn, acc):
    """online_targeting.run_kicks_online + accounting into `acc` (see A_* indices)."""
    cuts = np.empty(3, dtype=np.int64)
    rows = np.empty(3, dtype=np.int64)
    eu = np.empty(3, dtype=np.int64)
    ev = np.empty(3, dtype=np.int64)
    xsum = np.empty(X.shape[1])
    for _ in range(n_kicks):
        delta = _kick_global_cuts(tour, pos, n, xs, ys, kind, guided, pool_u, pool_v, pool_m,
                                  prob, queue, inq, qht, buf, stats, cuts)
        # --- accounting: rows are computed for both arms (no RNG involved) ---
        rowok = True
        for q in range(3):
            p = cuts[q]
            eu[q] = btour[p - 1]
            ev[q] = btour[p]
            rows[q] = _ref_row(btour[p - 1], btour[p], ref_tour, ref_pos, n)
            if rows[q] < 0:
                rowok = False
        ok = learn and rowok
        g = local_search(tour, pos, n, xs, ys, kind, nbr, queue, inq, qht, buf)
        newc = cost + delta - g
        stats[0] += 1
        # --- accounting ---
        r = -(newc - cost) * inv_scale
        acc[A_KICKS] += 1
        acc[A_SUMR] += r
        acc[A_SUMR2] += r * r
        acc[A_SUMDMG] += delta * inv_scale
        if newc < cost - EPS:
            acc[A_IMPR] += 1
            acc[A_SUMACC] += r
        if abs(r) < NEUTRAL_TOL:
            acc[A_NEUT] += 1
        elif newc > cost:
            acc[A_WORSE] += 1
        if rowok:
            acc[A_ROWOK] += 1
            acc[A_SUMR_ROWOK] += r
            if abs(r) < NEUTRAL_TOL:
                acc[A_NEUT_ROWOK] += 1
        nres = 0
        for q in range(3):        # are all three removed edges tour edges again after LS?
            i = pos[eu[q]]
            if tour[(i + 1) % n] == ev[q] or tour[(i - 1 + n) % n] == ev[q]:
                nres += 1
        if nres == 3:
            acc[A_RESTORED] += 1
        # --- original update (identical to run_kicks_online) ---
        if ok:
            for k in range(X.shape[1]):
                xsum[k] = X[rows[0], k] + X[rows[1], k] + X[rows[2], k]
            for k in range(X.shape[1]):
                bvec[k] += r * xsum[k]
                for l in range(X.shape[1]):
                    A[k, l] += xsum[k] * xsum[l]
            stats[4] += 1
        if newc < cost - EPS:
            cost = newc
            stats[1] += 1
            for i in range(n):
                btour[i] = tour[i]
                bpos[i] = pos[i]
        else:
            for i in range(n):
                tour[i] = btour[i]
                pos[i] = bpos[i]
    return cost


def run_probe(xs, ys, kind, nbr, init_tour, init_cost, seed, budget, knn24, params, w0=None,
              lam=50.0, nu=0.5, refresh_kicks=2000, prob=0.75, overhead_cap=0.10,
              batch_target_s=0.004, learn=True):
    """Mirror of online_targeting.run_online (same arguments / control flow) with the probe kernel."""
    n = len(xs)
    tour = init_tour.copy()
    pos = np.empty(n, dtype=np.int32)
    pos[tour] = np.arange(n, dtype=np.int32)
    btour = tour.copy()
    bpos = pos.copy()
    queue = np.empty(n, dtype=np.int32)
    inq = np.zeros(n, dtype=np.uint8)
    qht = np.zeros(3, dtype=np.int64)
    buf = np.empty(n, dtype=np.int32)
    stats = np.zeros(5, dtype=np.int64)
    F.seed_rng(seed)
    m = F.pool_size(n)
    pool_u = np.zeros(m, dtype=np.int32)
    pool_v = np.zeros(m, dtype=np.int32)
    pool_m = 0
    cost = init_cost
    checkpoints = [t for t in F.CHECKPOINTS if t <= budget + 1e-9]
    trace = {}
    model_time = 0.0
    refreshes = 0
    accepted_at_refresh = -1
    kicks_at_refresh = 0
    batch = 8
    pol = OT.OnlinePolicy(xs, ys, kind, knn24, params, w0, lam, nu, seed)
    X = ref_tour = ref_pos = None
    el0 = F._edge_lengths(np.ascontiguousarray(btour), np.ascontiguousarray(np.roll(btour, -1)), xs, ys, kind)
    med0 = max(float(np.median(el0)), 1e-12)
    inv_scale = 1.0 / med0
    b17_start = float(pol.b[D - 1])
    accs = np.zeros((len(WINDOWS) + 1, NACC))       # last row: beyond the last window (should stay 0)
    pool_len_ratio = []                              # mean pool edge length / median tour edge length
    start = time.perf_counter()
    elapsed = 0.0
    while True:
        elapsed = time.perf_counter() - start
        while checkpoints and elapsed >= checkpoints[0]:
            trace[str(checkpoints.pop(0))] = cost
        if elapsed >= budget:
            break
        due = stats[1] != accepted_at_refresh or stats[0] - kicks_at_refresh >= refresh_kicks
        if due and (refreshes == 0 or model_time <= overhead_cap * max(elapsed, 1e-9)):
            t1 = time.perf_counter()
            if stats[1] != accepted_at_refresh or refreshes == 0:
                X = pol.features(btour)
                ref_tour = btour.copy()
                ref_pos = bpos.copy()
            w = pol.sample_w()
            s = X[:, :D - 1] @ w[:D - 1]
            idx = np.argpartition(-s, m - 1)[:m]
            pool_u[:] = btour[idx]
            pool_v[:] = btour[(idx + 1) % n]
            pool_m = m
            refreshes += 1
            accepted_at_refresh = stats[1]
            kicks_at_refresh = stats[0]
            model_time += time.perf_counter() - t1
            # accounting (not charged to model_time, outside the timed decision)
            el = F._edge_lengths(np.ascontiguousarray(btour), np.ascontiguousarray(np.roll(btour, -1)),
                                 xs, ys, kind)
            pool_len_ratio.append(float(el[idx].mean() / max(np.median(el), 1e-12)))
        wi = next((i for i, (lo, hi) in enumerate(WINDOWS) if lo <= elapsed < hi), len(WINDOWS))
        tb = time.perf_counter()
        cost = run_kicks_probe(tour, pos, btour, bpos, cost, n, xs, ys, kind, nbr, batch,
                               True, pool_u, pool_v, pool_m, prob, queue, inq, qht, buf, stats,
                               X, ref_tour, ref_pos, inv_scale, pol.A, pol.b, learn, accs[wi])
        dt = time.perf_counter() - tb
        if dt < batch_target_s * 0.5:
            batch = min(batch * 2, 1 << 20)
        elif dt > batch_target_s * 2 and batch > 1:
            batch = max(1, batch // 2)
    exact = float(F.tour_cost(btour, xs, ys, kind))
    if abs(exact - cost) > 1e-6 * max(1.0, abs(exact)):
        raise RuntimeError(f"cost drift: tracked {cost} vs exact {exact}")
    if len(np.unique(btour)) != n:
        raise RuntimeError("invalid tour")
    w_final = np.linalg.solve(pol.A, pol.b)
    tot = accs.sum(0)
    return {"cost": exact, "kicks": int(stats[0]), "accepted": int(stats[1]),
            "learned_kicks": int(stats[4]), "refreshes": refreshes, "model_time": model_time,
            "acc_windows": accs.tolist(), "acc_total": tot.tolist(),
            # consistency check with the unmodified update rule: delta b[bias] = 3 r per learned kick
            "sum_r_from_b": (float(pol.b[D - 1]) - b17_start) / 3.0,
            "w_final": w_final.tolist(), "pool_len_ratio_mean": float(np.mean(pool_len_ratio)),
            "pool_len_ratio_last": float(pool_len_ratio[-1]), "median_edge0": med0}


def summarize_acc(a):
    a = np.asarray(a)
    k = max(a[A_KICKS], 1.0)
    ro = max(a[A_ROWOK], 1.0)
    return {"kicks": int(a[A_KICKS]),
            "mean_r": a[A_SUMR] / k,
            "mean_r_learnable": a[A_SUMR_ROWOK] / ro,
            "frac_improving": a[A_IMPR] / k,
            "frac_neutral": a[A_NEUT] / k,
            "frac_worsening": a[A_WORSE] / k,
            "frac_all3_cut_edges_restored": a[A_RESTORED] / k,
            "mean_kick_damage_preLS": a[A_SUMDMG] / k,
            "mean_r_given_improving": a[A_SUMACC] / max(a[A_IMPR], 1.0),
            "mean_r_given_worsening": (a[A_SUMR] - a[A_SUMACC]) / max(a[A_WORSE], 1.0),
            "learnable_frac": a[A_ROWOK] / k}


def dev_instances():
    from kick_targeting_experiment import suite_instances
    rows = suite_instances("vlsi_primary")
    idx = np.unique(np.round(np.linspace(0, len(rows) - 1, 12)).astype(int))
    return [rows[i] for i in idx]


def cmd_run(seeds=3, budget=10.0):
    from cheap_scorer import MODEL, load_params
    from kick_targeting_experiment import load_xy
    params = load_params(MODEL)
    w0 = OT.load_prior()
    OT.warmup(params)
    # compile the probe kernel on a tiny instance (both arms) so timed runs exclude JIT
    rng = np.random.default_rng(0)
    for kind in (F.EUC, F.GEO):
        xs, ys = rng.uniform(30, 50, 80), rng.uniform(0, 20, 80)
        nb, k24 = F.build_knn(xs, ys, kind, 8), F.build_knn(xs, ys, kind, 24)
        t0, c0, _ = F.start_tour(xs, ys, kind, nb, 0)
        run_probe(xs, ys, kind, nb, t0, c0, 0, 0.05, k24, params, w0=w0)
    arms = {"onlinewarm": {"w0": w0}, "linprior": {"w0": w0, "learn": False, "nu": 0.0}}
    insts = dev_instances()
    print("instances:", [r[0] for r in insts], flush=True)
    records = []
    t_all = time.time()
    for name, path, bks in insts:
        xs, ys, kind = load_xy(path)
        n = len(xs)
        nbr = F.build_knn(xs, ys, kind, 10)
        knn24 = F.build_knn(xs, ys, kind, min(24, n - 1))
        for seed in range(seeds):
            init, c0, _ = F.start_tour(xs, ys, kind, nbr, 7919 * (seed + 1) + n)
            kick_seed = 104729 + 1009 * seed
            order = list(arms)
            order = order[seed % 2:] + order[:seed % 2]
            rec = {"instance": name, "n": n, "bks": bks, "seed": seed, "arms": {}}
            for arm in order:
                res = run_probe(xs, ys, kind, nbr, init, c0, kick_seed, budget, knn24, params, **arms[arm])
                res["gap_pct"] = 100.0 * (res["cost"] - bks) / bks
                res["summary"] = summarize_acc(res["acc_total"])
                res["summary_windows"] = {f"{lo:g}-{hi:g}s": summarize_acc(res["acc_windows"][i])
                                          for i, (lo, hi) in enumerate(WINDOWS)}
                w = np.array(res["w_final"])
                res["cos_w0_wfinal"] = float(w @ w0 / (np.linalg.norm(w) * np.linalg.norm(w0)))
                rec["arms"][arm] = res
            records.append(rec)
            o, s = rec["arms"]["onlinewarm"], rec["arms"]["linprior"]
            print(f"[{time.time() - t_all:6.0f}s] {name} s{seed}  gap on/st={o['gap_pct']:.3f}/{s['gap_pct']:.3f}  "
                  f"mean r on/st={o['summary']['mean_r']:+.4f}/{s['summary']['mean_r']:+.4f}  "
                  f"neutral on/st={o['summary']['frac_neutral']:.3f}/{s['summary']['frac_neutral']:.3f}  "
                  f"impr on/st={o['summary']['frac_improving']:.4f}/{s['summary']['frac_improving']:.4f}  "
                  f"b-check {o['sum_r_from_b']:.3f} vs {o['acc_total'][A_SUMR_ROWOK]:.3f}", flush=True)
    out = aggregate(records)
    out.update({"description": __doc__.strip().splitlines()[0], "budget_s": budget, "seeds": seeds,
                "neutral_tol_r": NEUTRAL_TOL, "windows_s": WINDOWS,
                "arms": {"onlinewarm": "run_online(w0=prior, lam 50, nu 0.5, refresh 2000, pool 0.06/0.75), learns",
                         "linprior": "same, learn=False, nu=0 (static prior), as in PROTOCOL_v2"},
                "wall_s": time.time() - t_all, "records": records})
    p = OUT / "online_reward_mechanism.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("written", p)
    print(json.dumps({k: v for k, v in out.items() if k not in ("records",)}, indent=1))


def _paired(vals_a, vals_b):
    from scipy import stats
    d = np.asarray(vals_a) - np.asarray(vals_b)
    p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"mean_online": float(np.mean(vals_a)), "mean_static": float(np.mean(vals_b)),
            "mean_diff": float(d.mean()), "online_higher_count": int((d > 0).sum()),
            "n": int(len(d)), "p_wilcoxon": p}


def aggregate(records):
    """Instance = unit (seeds averaged).  Keys compared online vs static."""
    from collections import defaultdict
    keys = ["mean_r", "mean_r_learnable", "frac_improving", "frac_neutral", "frac_worsening",
            "frac_all3_cut_edges_restored", "mean_kick_damage_preLS", "mean_r_given_improving",
            "mean_r_given_worsening"]
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for rec in records:
        for arm, res in rec["arms"].items():
            per[rec["instance"]][arm]["gap_pct"].append(res["gap_pct"])
            per[rec["instance"]][arm]["pool_len_ratio_mean"].append(res["pool_len_ratio_mean"])
            per[rec["instance"]][arm]["cos_w0_wfinal"].append(res["cos_w0_wfinal"])
            for k in keys:
                per[rec["instance"]][arm][k].append(res["summary"][k])
            for wn, sw in res["summary_windows"].items():
                per[rec["instance"]][arm][f"mean_r@{wn}"].append(sw["mean_r"])
                per[rec["instance"]][arm][f"frac_neutral@{wn}"].append(sw["frac_neutral"])
    inst = sorted(per)
    allkeys = ["gap_pct", "pool_len_ratio_mean", "cos_w0_wfinal"] + keys + \
        [f"{m}@{lo:g}-{hi:g}s" for m in ("mean_r", "frac_neutral") for lo, hi in WINDOWS]
    comp = {}
    for k in allkeys:
        a = [np.mean(per[i]["onlinewarm"][k]) for i in inst]
        b = [np.mean(per[i]["linprior"][k]) for i in inst]
        comp[k] = _paired(a, b)
    b_check = max(abs(r["arms"]["onlinewarm"]["sum_r_from_b"] - r["arms"]["onlinewarm"]["acc_total"][A_SUMR_ROWOK])
                  / max(1.0, abs(r["arms"]["onlinewarm"]["acc_total"][A_SUMR_ROWOK])) for r in records)
    return {"instances": inst, "comparison_online_minus_static": comp,
            "max_rel_err_sum_r_vs_delta_b_bias_over_3": float(b_check)}


# ---------------------------------------------------------------------------
# Task 3: Hodges-Lehmann estimates for the confirmatory hypotheses (no new runs)
# ---------------------------------------------------------------------------


def hodges_lehmann(d, alpha=0.05):
    """HL pseudo-median of d (median of Walsh averages) with the exact distribution-free CI
    from the signed-rank distribution (Hollander & Wolfe): order statistics W_(k), W_(M+1-k)
    of the M = n(n+1)/2 Walsh averages, k = largest integer with P(T+ <= k-1) <= alpha/2."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    i, j = np.triu_indices(n)
    W = np.sort(0.5 * (d[i] + d[j]))
    M = len(W)
    # exact null distribution of the Wilcoxon signed-rank statistic T+ (counts over 2^n)
    cnt = np.zeros(M + 1, dtype=object)
    cnt[0] = 1
    for r in range(1, n + 1):
        nxt = cnt.copy()
        nxt[r:] = nxt[r:] + cnt[:M + 1 - r]
        cnt = nxt
    cdf = np.cumsum(cnt) / float(2 ** n)
    cdf = np.array(cdf, dtype=float)
    # k: P(T+ <= k-1) <= alpha/2
    k = int(np.searchsorted(cdf, alpha / 2, side="right"))  # number of t with cdf(t) <= alpha/2
    k = max(k, 1)
    lo, hi = W[k - 1], W[M - k]
    coverage = 1 - 2 * cdf[k - 1]
    return float(np.median(W)), [float(lo), float(hi)], float(coverage), k


def cmd_hl(tag="holdout2_10s", boot=20000, seed=0):
    import v2_analysis as VA
    proto = json.loads((OUT / "PROTOCOL_v2.json").read_text(encoding="utf-8"))
    data, _ = VA.per_instance(tag)
    rng = np.random.default_rng(seed)
    rows = []
    for h in proto["confirmatory_hypotheses"]:
        keys = sorted(k for k in data if h["a"] in data[k] and h["b"] in data[k])
        d = np.array([data[k][h["a"]] - data[k][h["b"]] for k in keys])
        hl, ci, cov, k = hodges_lehmann(d)
        bs_mean = d[rng.integers(len(d), size=(boot, len(d)))].mean(1)
        # bootstrap of the HL estimator as a cross-check of the exact interval
        bs_hl = []
        for _ in range(2000):
            s = d[rng.integers(len(d), size=len(d))]
            ii, jj = np.triu_indices(len(s))
            bs_hl.append(np.median(0.5 * (s[ii] + s[jj])))
        rows.append({"id": h["id"], "a": h["a"], "b": h["b"], "test": h["test"], "margin": h.get("margin"),
                     "n_inst": int(len(d)), "mean": float(d.mean()), "median": float(np.median(d)),
                     "hl_pseudomedian": hl, "hl_ci95_exact": ci, "hl_ci_exact_coverage": cov,
                     "hl_ci_rank_k": k,
                     "hl_ci95_bootstrap_percentile": [float(np.quantile(bs_hl, 0.025)),
                                                      float(np.quantile(bs_hl, 0.975))],
                     "mean_ci95_bootstrap_percentile": [float(np.quantile(bs_mean, 0.025)),
                                                        float(np.quantile(bs_mean, 0.975))],
                     "mean_upper95_one_sided_bootstrap": float(np.quantile(bs_mean, 0.95))})
        r = rows[-1]
        print(f"{r['id']}: {r['a']} - {r['b']}  n={r['n_inst']}  mean={r['mean']:+.4f} "
              f"bootCI[{r['mean_ci95_bootstrap_percentile'][0]:+.4f},{r['mean_ci95_bootstrap_percentile'][1]:+.4f}]  "
              f"HL={hl:+.4f} exactCI[{ci[0]:+.4f},{ci[1]:+.4f}] (cov {cov:.4f})  margin={r['margin']}")
    out = {"tag": tag, "unit": "instance (seeds averaged), difference = a - b in gap points, negative = a better",
           "method": "Hodges-Lehmann = median of the n(n+1)/2 Walsh averages; exact CI from the null distribution "
                     "of the signed-rank statistic; bootstrap: percentile, instance resampling "
                     f"({boot} reps for the mean, 2000 for HL), rng seed {seed}",
           "hypotheses": rows}
    p = OUT / f"{tag}_hl.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("written", p)


if __name__ == "__main__":
    {"run": cmd_run, "hl": cmd_hl}[sys.argv[1]]()

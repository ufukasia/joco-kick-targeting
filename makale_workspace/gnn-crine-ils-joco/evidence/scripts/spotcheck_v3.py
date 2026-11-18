# Independent recomputation from raw run files (no v2_analysis import) vs evidence.json records.
import glob, json, os, statistics
from collections import defaultdict
from scipy import stats

ROOT = r"C:\Users\ufuka\OneDrive - Ostim Teknik Universitesi\Masaüstü\çöp\tsplib"
WS = os.path.join(ROOT, "makale_workspace", "gnn-crine-ils-joco")
ev = json.load(open(os.path.join(WS, "evidence", "evidence.json"), encoding="utf-8"))
K = {r["ek"]["anahtar"]: r for r in ev["kayitlar"] if isinstance(r.get("ek"), dict) and "anahtar" in r["ek"]}


def inst_means(tag):
    acc = defaultdict(lambda: defaultdict(list))
    raw = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(os.path.join(ROOT, "results", "kick_targeting", tag, "*__s*.json")):
        d = json.load(open(f, encoding="utf-8"))
        for s, r in d["runs"].items():
            acc[d["instance"]][s].append(r["gap_pct"])
            raw[s]["mshare"].append(r["model_time"] / r["elapsed"])
            raw[s]["stale"].append(r["stale_cuts"] / max(r["guided_cuts"], 1))
            raw[s]["kps"].append(r["kicks"] / r["elapsed"])
    return {i: {s: sum(v) / len(v) for s, v in d.items()} for i, d in acc.items()}, raw


def chk(name, mine, key, field="deger", tol=1e-9):
    rec = K[key]
    val = rec[field] if field == "deger" else rec["ek"][field]
    ok = abs(mine - val) <= tol * max(1, abs(mine))
    print(f"{'OK ' if ok else 'XX '} {name:45s} indep={mine:.6g}  evidence[{rec['id']}]={val:.6g}")
    return ok


h, raw = inst_means("holdout2_10s")
cat = json.load(open(os.path.join(ROOT, "data_holdout2", "catalog.json")))["instances"]
E, R, G, KN = "excess_global@0.06:0.5", "random_global", "gnn_global@0.04:0.5", "knn_global@0.04:0.75"
d_er = [h[i][E] - h[i][R] for i in h]
d_eg = [h[i][E] - h[i][G] for i in h]
res = []
res.append(chk("V2 mean excess-random", statistics.fmean(d_er), "v3.confirm.V2"))
res.append(chk("V2 p (two-sided Wilcoxon)", stats.wilcoxon(d_er).pvalue, "v3.confirm.V2", "p", 1e-6))
res.append(chk("V3 NI p (d-0.05, less)", stats.wilcoxon([x - 0.05 for x in d_eg], alternative="less").pvalue,
               "v3.confirm.V3", "p", 1e-6))
res.append(chk("excess vs GNN wins", sum(x < 0 for x in d_eg), "v3.confirm.V3", "wins_a"))
res.append(chk("excess mean gap (all 54)", statistics.fmean(h[i][E] for i in h), "v3.h2.gap.all.excess"))
cl = [i for i in h if cat[i]["stratum"] == "synth_clustered"]
res.append(chk("excess mean gap (clustered 14)", statistics.fmean(h[i][E] for i in cl), "v3.h2.gap.clustered.excess"))
un = [i for i in h if cat[i]["stratum"] == "synth_uniform"]
res.append(chk("uniform excess-random p", stats.wilcoxon([h[i][E] - h[i][R] for i in un]).pvalue,
               "v3.h2.gap.uniform.excess", "p_wilcoxon", 1e-6))
res.append(chk("GNN model share", statistics.fmean(raw[G]["mshare"]), "v3.mech.model_share.gnn"))
res.append(chk("GNN stale share", statistics.fmean(raw[G]["stale"]), "v3.mech.stale_share.gnn"))
res.append(chk("online kicks/s", statistics.fmean(raw["onlinewarm_global"]["kps"]), "v3.mech.kicks_per_s.onlinewarm"))
# Holm (5-pair family): excess-knn raw p * 5 (smallest)
p_ek = stats.wilcoxon([h[i][E] - h[i][KN] for i in h]).pvalue
res.append(chk("Holm p excess-knn (rank 1 of 5)", 5 * p_ek, "v3.secondary.excess-knn", "p_holm", 1e-6))
# development
dv, _ = inst_means("dev_vlsi_primary_10s")
dd = [dv[i]["onlinewarm_global"] - dv[i]["linprior_global"] for i in dv]
res.append(chk("dev VLSI-38 online-linprior mean", statistics.fmean(dd), "v3.dev.dev_vlsi_primary_10s.onlinewarm_global-linprior_global"))
# grid: excess 0.06:0.5 minus random, equal-weight over two sets
vals = []
for tag in ("dev_grid_vlsi_primary_10s", "dev_grid_synth1000_10s"):
    g, _ = inst_means(tag)
    vals.append(statistics.fmean(g[i]["excess_global@0.06:0.5"] for i in g) - statistics.fmean(g[i][R] for i in g))
res.append(chk("grid excess@0.06:0.5 delta vs random", statistics.fmean(vals), "v3.grid.cell.excess_global@0.06:0.5"))
# drift mean from mechanism json array
m = json.load(open(os.path.join(ROOT, "results", "kick_targeting", "holdout2_10s_mechanism.json")))
res.append(chk("drift rel_len_pool_learned mean", statistics.fmean(r["rel_len_pool_learned"] for r in m["drift"]),
               "v3.drift.rel_len_pool_learned"))
print(f"\n{sum(res)}/{len(res)} spot checks passed")

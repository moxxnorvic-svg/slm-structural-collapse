#!/usr/bin/env python3
"""Task-level re-analysis of experiment_results*.json for the SCP paper (reviewer response).
Usage: python analyze_scp.py experiment_results.json
Requires: pandas, numpy, scipy, statsmodels, lifelines
Writes: table1_pass_rates.csv, table3_collapse.csv, table4_tests.csv,
        table5_sensitivity.csv, table6_errors.csv
"""
import sys, json
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import beta
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from lifelines.statistics import logrank_test

df = pd.DataFrame(json.load(open(sys.argv[1] if len(sys.argv) > 1 else "experiment_results.json")))
df["m"] = df.model.map({"qwen2.5-coder:3b": "Qwen", "llama3.2:3b": "Llama"})
df["a"] = df.arm.map({"Baseline_Greedy": "Base", "Isolated_Self_Critique": "Self",
                      "Compiler_Grounded_Feedback": "Comp"})
df = df.sort_values(["m", "a", "task_id", "iteration"])
traj = {k: g for k, g in df.groupby(["m", "a", "task_id"])}

def wilson(k, n=40, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d
    h = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return f"[{100*(c-h):.1f}, {100*(c+h):.1f}]"

# --- SCP definitions -------------------------------------------------------
def scp_first_failure(g):          # identical to calculate_scp() in run_eval.py
    it, sv, p, ted = g.iteration.tolist(), g.SV.tolist(), g.PassRate.tolist(), g.TED.tolist()
    for i in range(len(sv) - 2):
        if sv[i] == sv[i+1] == sv[i+2] == 0: return it[i]
    if 0 in p:
        f = p.index(0)
        if f == 0: return it[0]
        d = np.diff(np.array(ted, float)); s = max(0, f-1); e = min(len(d)-1, f+1)
        mi = s + int(np.argmax(np.abs(d[s:e+1]))); return it[min(mi+1, len(it)-1)]
    return None

def scp_eq4(g, eps=0.0, w=3):      # Eq. 4: w-turn window, TED = consecutive-turn raw nodes
    p, sv, ted = g.PassRate.values, g.SV.values, g.TED.values
    for t in range(len(p) - w + 1):
        win = range(t, t + w)
        if all(p[k] == 0 and ted[k] <= eps for k in win) or all(sv[k] == 0 for k in win):
            return t + 1
    return None

def scp_never(g):
    p = g.PassRate.values
    for t in range(len(p)):
        if p[t:].sum() == 0: return t + 1
    return None

# --- Table 1 ---------------------------------------------------------------
t1 = df.pivot_table(index=["m", "a"], columns="iteration", values="PassRate", aggfunc="sum")
t1.to_csv("table1_pass_rates.csv")

# --- per-task summary ------------------------------------------------------
rows = []
for (m, a, t), g in traj.items():
    p = g.PassRate.values
    ff = scp_first_failure(g)
    rec = int(p[np.argmax(p == 0):].max() == 1) if (p == 0).any() else None
    rows.append(dict(m=m, a=a, task=t, first=ff, eq4=scp_eq4(g), never=scp_never(g),
                     recovered=rec, nfail=int((p == 0).sum()), meanTED=g.TED.mean(), final=int(p[-1])))
r = pd.DataFrame(rows)
r["coll"] = r["first"].notna().astype(int)

t3 = []
for (m, a), g in r.groupby(["m", "a"]):
    s = g["first"].dropna(); k = len(s)
    t3.append(dict(model=m, arm=a, first_failure=k, pct=100*k/40, ci=wilson(k),
                   by_t5=int((s <= 5).sum()), median=s.median(), q1=s.quantile(.25), q3=s.quantile(.75),
                   recovered=int(g.recovered.sum()), persistent_eq4=int(g.eq4.notna().sum())))
pd.DataFrame(t3).to_csv("table3_collapse.csv", index=False)

# --- Table 4 ---------------------------------------------------------------
def v(m, a, col): return r[(r.m == m) & (r.a == a)].sort_values("task")[col].values
tests = []
def mc(x, y, lab):
    b = int(((x == 1) & (y == 0)).sum()); c = int(((x == 0) & (y == 1)).sum())
    p = mcnemar([[0, b], [c, 0]], exact=True).pvalue
    lo = beta.ppf(.025, b, c+1) if b else 0; hi = beta.ppf(.975, b+1, c) if c else 1
    odds = lambda q: q/(1-q) if q < 1 else np.inf
    tests.append(dict(test=lab, method="McNemar exact", stat=f"b={b}, c={c}", p=p,
                      effect=f"OR={b/c if c else np.inf:.2f} [{odds(lo):.2f}, {odds(hi):.2f}]"))
def wil(x, y, lab):
    d = x - y; nz = d[d != 0]; rk = stats.rankdata(np.abs(nz))
    wp, wm = rk[nz > 0].sum(), rk[nz < 0].sum()
    tests.append(dict(test=lab, method="Wilcoxon signed-rank", stat=f"W={min(wp, wm):.1f} (n={len(nz)})",
                      p=stats.wilcoxon(x, y).pvalue, effect=f"r_rb={(wp-wm)/(wp+wm):.2f}"))
for m in ["Qwen", "Llama"]: mc(v(m, "Comp", "coll"), v(m, "Self", "coll"), f"{m}: Comp vs Self, first failure")
mc(v("Qwen", "Comp", "coll"), v("Llama", "Comp", "coll"), "Comp: Qwen vs Llama, first failure")
for m in ["Qwen", "Llama"]: wil(v(m, "Comp", "nfail"), v(m, "Self", "nfail"), f"{m}: Comp vs Self, failing turns")
for m in ["Qwen", "Llama"]: wil(v(m, "Comp", "meanTED"), v(m, "Self", "meanTED"), f"{m}: Comp vs Self, mean TED")
for a in ["Self", "Comp"]: wil(v("Qwen", a, "meanTED"), v("Llama", a, "meanTED"), f"{a}: Qwen vs Llama, mean TED")
for m in ["Qwen", "Llama"]:
    A, B = r[(r.m == m) & (r.a == "Comp")], r[(r.m == m) & (r.a == "Self")]
    lr = logrank_test(A["first"].fillna(15), B["first"].fillna(15), A["first"].notna(), B["first"].notna())
    tests.append(dict(test=f"{m}: Comp vs Self, time to first failure", method="log-rank",
                      stat=f"chi2={lr.test_statistic:.2f}", p=lr.p_value, effect=""))
t4 = pd.DataFrame(tests); t4["p_holm"] = multipletests(t4.p, method="holm")[1]
t4.to_csv("table4_tests.csv", index=False)

# --- Table 5 sensitivity ---------------------------------------------------
sens = []
defs = {f"eq4 eps={e} w={w}": (lambda g, e=e, w=w: scp_eq4(g, e, w))
        for e, w in [(0, 3), (1, 3), (2, 3), (5, 3), (0, 2), (0, 4)]}
defs["never passes again"] = scp_never; defs["first failure"] = scp_first_failure
for name, f in defs.items():
    vals = {(m, a): [f(g) for (mm, aa, t), g in traj.items() if mm == m and aa == a]
            for m in ["Llama", "Qwen"] for a in ["Self", "Comp"]}
    allv = [x for (m, a, t), g in traj.items() for x in [f(g)] if x is not None]
    row = {"definition": name}
    row.update({f"{m}_{a}": sum(x is not None for x in xs) for (m, a), xs in vals.items()})
    row["by_t5"] = f"{sum(x <= 5 for x in allv)}/{len(allv)}"
    sens.append(row)
pd.DataFrame(sens).to_csv("table5_sensitivity.csv", index=False)

# --- Table 6 errors --------------------------------------------------------
df["err"] = df.error_log.str.split(":").str[0]
df[df.err != "Functional Success"].pivot_table(index=["m", "a"], columns="err", values="iteration",
    aggfunc="count", fill_value=0, margins=True).to_csv("table6_errors.csv")

print(t4.to_string(index=False)); print(pd.DataFrame(sens).to_string(index=False))

"""Figures and worked-example numbers for 'Evaluating Large Language Models in High-Stakes Domains'.
All data are analytic or simulated; generating processes are stated in the text.
Run:  python figures.py   (writes PNGs and results.json into ./figures)
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize, minimize_scalar

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

# palette (categorical, fixed order) and ink
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "figure.dpi": 200, "savefig.dpi": 200,
    "savefig.facecolor": "white",
})
results = {}
se = lambda x: x.std(ddof=1)/np.sqrt(len(x))
def wilson(x, n, z=1.96):
    p = x/n; c = (p + z*z/(2*n))/(1 + z*z/n)
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/(1 + z*z/n)
    return c-h, c+h
def clopper_upper(x, n, conf=0.95):
    return 1.0 if x == n else stats.beta.ppf(conf, x+1, n-x)

# ---------------------------------------------------------------- Figure 1
def wilson_halfwidth(p, n, z=1.96):
    return z*np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / (1 + z**2/n)

n = np.arange(20, 1501)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
for p, c in zip([0.70, 0.85, 0.95], [C1, C2, C3]):
    ax[0].plot(n, 100*wilson_halfwidth(p, n), color=c, lw=2, label=f"accuracy {p:.2f}")
ax[0].set_xlabel("Number of items n"); ax[0].set_ylabel("95% CI half-width (points)")
ax[0].set_title("(a) Precision of a single accuracy", loc="left", fontsize=9, color=INK)
ax[0].set_xlim(0, 1550); ax[0].legend(loc="upper right")
zs = 1.96 + 0.8416
for disc, c in zip([0.10, 0.20, 0.30], [C1, C2, C3]):
    d = np.sqrt(zs**2*disc/(n + zs**2))
    ax[1].plot(n, 100*d, color=c, lw=2, label=f"discordance {disc:.2f}")
ax[1].set_xlabel("Number of items n"); ax[1].set_ylabel("Minimum detectable difference (points)")
ax[1].set_title("(b) Paired comparison, 80% power", loc="left", fontsize=9, color=INK)
ax[1].set_xlim(0, 1550); ax[1].legend(loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_ci_width.png")); plt.close(fig)

# ---------------------------------------------------------------- Worked example, Ch. 2
rng = np.random.default_rng(7)
n_, k = 120, 5
item_effect = rng.normal(0, 1.6, n_)
logit_A = 2.0 + item_effect + rng.normal(0, .3, n_)
logit_B = 1.4 + item_effect + rng.normal(0, .3, n_)
pA, pB = 1/(1+np.exp(-logit_A)), 1/(1+np.exp(-logit_B))
sA = rng.binomial(1, pA[:, None], (n_, k)); sB = rng.binomial(1, pB[:, None], (n_, k))
mA, mB = sA.mean(1), sB.mean(1)
d = mA - mB
B = 5000; idx = rng.integers(0, n_, (B, n_)); boot = (mA[idx]-mB[idx]).mean(1)
lo, hi = np.percentile(boot, [2.5, 97.5])
results["ch2"] = {
    "A_ACC": f"{mA.mean():.3f} [{mA.mean()-1.96*se(mA):.3f}, {mA.mean()+1.96*se(mA):.3f}]",
    "B_ACC": f"{mB.mean():.3f} [{mB.mean()-1.96*se(mB):.3f}, {mB.mean()+1.96*se(mB):.3f}]",
    "A_NAIVE": f"{se(sA.ravel()):.4f}", "A_CLUST": f"{se(mA):.4f}",
    "PAIRED": f"{d.mean():.3f} [{d.mean()-1.96*se(d):.3f}, {d.mean()+1.96*se(d):.3f}]",
    "UNPAIRED_SE": f"{np.sqrt(se(mA)**2+se(mB)**2):.4f}", "PAIRED_SE": f"{se(d):.4f}",
    "BOOT": f"[{lo:.3f}, {hi:.3f}]",
}

# ---------------------------------------------------------------- Worked example, Ch. 3
rng = np.random.default_rng(33)
N3 = 3000                                   # outputs scored by the judge
q = rng.beta(5, 2, N3)                      # latent quality of each output
s_true = np.clip(q + rng.normal(0, 0.05, N3), 0, 1)   # expert ("true") rubric score
# (a) three human raters on a random subset of 150 outputs: s_ir = s_i + b_r + e_ir
n_r, R = 150, 3
sub = rng.choice(N3, n_r, replace=False)
b_r = np.array([0.0, 0.06, -0.03])
ratings = s_true[sub][:, None] + b_r[None, :] + rng.normal(0, 0.10, (n_r, R))
ratings = np.clip(ratings, 0, 1)
def kripp_alpha_interval(M):             # M: items x raters, no missing
    n_i, R_ = M.shape
    Do = np.mean([np.sum((M[i][:, None]-M[i][None, :])**2)/(R_-1) for i in range(n_i)])/R_
    pooled = M.ravel(); n_p = len(pooled)
    De = np.sum((pooled[:, None]-pooled[None, :])**2)/(n_p*(n_p-1))
    return 1 - Do/De
alpha_hat = kripp_alpha_interval(ratings)
bootA = np.array([kripp_alpha_interval(ratings[rng.integers(0, n_r, n_r)]) for _ in range(2000)])
# generalizability theory: two-way random effects (items x raters), method of moments
grand = ratings.mean(); mi = ratings.mean(1); mr = ratings.mean(0)
SS_i = R*np.sum((mi-grand)**2); SS_r = n_r*np.sum((mr-grand)**2)
SS_e = np.sum((ratings - mi[:, None] - mr[None, :] + grand)**2)
MS_i, MS_r, MS_e = SS_i/(n_r-1), SS_r/(R-1), SS_e/((n_r-1)*(R-1))
var_e = MS_e; var_i = max((MS_i-MS_e)/R, 0); var_r = max((MS_r-MS_e)/n_r, 0)
se_items_only = np.sqrt(var_i/n_r + var_e/(n_r*R))        # what the Ch.2 SE across item means gives
se_full = np.sqrt(var_i/n_r + var_r/R + var_e/(n_r*R))    # adds the rater facet
se_full_R5 = np.sqrt(var_i/n_r + var_r/5 + var_e/(n_r*5))
se_full_n300 = np.sqrt(var_i/300 + var_r/R + var_e/(300*R))
results["ch3"] = {
    "ALPHA": f"{alpha_hat:.3f}", "ALPHA_CI": f"[{np.percentile(bootA, 2.5):.3f}, {np.percentile(bootA, 97.5):.3f}]",
    "VAR_I": f"{var_i:.4f}", "VAR_R": f"{var_r:.4f}", "VAR_E": f"{var_e:.4f}",
    "SE_ITEMS": f"{se_items_only:.4f}", "SE_FULL": f"{se_full:.4f}",
    "SE_R5": f"{se_full_R5:.4f}", "SE_N300": f"{se_full_n300:.4f}",
    "RATER_SD": f"{np.sqrt(var_r):.3f}", "PCT_UNDER": f"{100*(1-se_items_only/se_full):.0f}",
}
# (b) judge validation for the unsafe category
p_unsafe = 1/(1+np.exp(-(-2.6 - 3.0*(q-0.6))))         # unsafe more likely for low-quality outputs
unsafe = rng.binomial(1, p_unsafe)                       # expert label
sens_true, spec_true = 0.60, 0.97
judge_unsafe = np.where(unsafe == 1, rng.binomial(1, sens_true, N3), rng.binomial(1, 1-spec_true, N3))
rand100 = rng.choice(N3, 100, replace=False)
u_r = unsafe[rand100]; j_r = judge_unsafe[rand100]
tp_r = int(((u_r==1)&(j_r==1)).sum()); pos_r = int((u_r==1).sum())
enr = np.concatenate([rng.choice(np.where(unsafe==1)[0], 50, replace=False),
                      rng.choice(np.where(unsafe==0)[0], 50, replace=False)])
u_e = unsafe[enr]; j_e = judge_unsafe[enr]
tp_e = int(((u_e==1)&(j_e==1)).sum()); pos_e = int((u_e==1).sum())
tn_e = int(((u_e==0)&(j_e==0)).sum()); neg_e = int((u_e==0).sum())
# agreement on the random sample vs sensitivity
agree_r = float((u_r == j_r).mean())
lo_r, hi_r = wilson(tp_r, pos_r) if pos_r else (np.nan, np.nan)
lo_e, hi_e = wilson(tp_e, pos_e); lo_s, hi_s = wilson(tn_e, neg_e)
results["ch3"].update({
    "UNSAFE_RATE": f"{100*unsafe.mean():.1f}", "AGREE_R": f"{100*agree_r:.0f}",
    "POS_R": str(pos_r), "SENS_R": f"{tp_r/pos_r:.2f} [{lo_r:.2f}, {hi_r:.2f}]" if pos_r else "n/a",
    "SENS_E": f"{tp_e/pos_e:.2f} [{lo_e:.2f}, {hi_e:.2f}]", "SPEC_E": f"{tn_e/neg_e:.2f} [{lo_s:.2f}, {hi_s:.2f}]",
})
# (c) prediction-powered inference with a lenient judge
judge_score = np.clip(s_true + 0.06 + rng.normal(0, 0.15, N3), 0, 1)
lab = rng.choice(N3, 300, replace=False); unl = np.setdiff1d(np.arange(N3), lab)
s_lab, sh_lab, sh_unl = s_true[lab], judge_score[lab], judge_score[unl]
n_l, N_u = len(s_lab), len(sh_unl)
v_lab = np.var(sh_lab, ddof=1); c = np.cov(s_lab, sh_lab, ddof=1)[0, 1]
lam = float(np.clip(c/(v_lab*(1 + n_l/N_u)), 0, 1))
est_pp = lam*sh_unl.mean() + (s_lab - lam*sh_lab).mean()
var_pp = lam**2*np.var(sh_unl, ddof=1)/N_u + np.var(s_lab - lam*sh_lab, ddof=1)/n_l
half_pp = 1.96*np.sqrt(var_pp); half_cl = 1.96*se(s_lab)
results["ch3"].update({
    "TRUE_MEAN": f"{s_true.mean():.3f}", "SD_S": f"{s_true.std(ddof=1):.2f}",
    "JUDGE_MEAN": f"{judge_score.mean():.3f}", "JUDGE_BIAS": f"{(judge_score-s_true).mean():+.3f}",
    "ERR_SD": f"{(judge_score-s_true).std(ddof=1):.2f}",
    "CLASSIC": f"{s_lab.mean():.3f} ± {half_cl:.3f}", "PPI": f"{est_pp:.3f} ± {half_pp:.3f}",
    "LAMBDA": f"{lam:.2f}", "PPI_RED": f"{100*(1-half_pp/half_cl):.0f}",
})

# ---------------------------------------------------------------- Figure 2: reliability diagram (Platt scaling)
rng = np.random.default_rng(11)
N = 4000
true_p = rng.beta(5, 2, N)
correct = rng.binomial(1, true_p)
conf = np.clip(0.55 + 0.6*(true_p-0.5) + rng.normal(0, 0.08, N), 0.05, 0.99)
conf = np.minimum(1.0, conf*1.18)
conf = np.round(conf*20)/20
bins = np.linspace(0, 1, 11)
def reliability(cf, y):
    w = np.clip(np.digitize(cf, bins)-1, 0, 9)
    acc = np.array([y[w==b].mean() if (w==b).any() else np.nan for b in range(10)])
    cfm = np.array([cf[w==b].mean() if (w==b).any() else np.nan for b in range(10)])
    cnt = np.array([(w==b).sum() for b in range(10)])
    return acc, cfm, cnt, np.nansum(cnt/len(y)*np.abs(acc-cfm))
acc_b, conf_b, cnt_b, ece = reliability(conf, correct)
logit = lambda p: np.log(p/(1-p))
half = rng.permutation(N); fit, ev = half[:N//2], half[N//2:]
zf = logit(np.clip(conf[fit], 1e-3, 1-1e-3))
def nll(ab):
    qf = 1/(1+np.exp(-(ab[0]*zf + ab[1])))
    return -np.mean(correct[fit]*np.log(qf)+(1-correct[fit])*np.log(1-qf))
a_pl, b_pl = minimize(nll, x0=[1.0, 0.0]).x
q_ev = 1/(1+np.exp(-(a_pl*logit(np.clip(conf[ev],1e-3,1-1e-3)) + b_pl)))
acc2, conf2, cnt2, ece2 = reliability(q_ev, correct[ev])
results["ch4"] = {"ECE_RAW": f"{ece:.3f}", "ECE_PL": f"{ece2:.3f}", "PLATT_A": f"{a_pl:.2f}", "PLATT_B": f"{b_pl:+.2f}",
                  "BRIER_RAW": f"{np.mean((conf-correct)**2):.3f}", "BRIER_PL": f"{np.mean((q_ev-correct[ev])**2):.3f}"}
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.1, 1]})
ax[0].plot([0,1],[0,1], ls="--", color=INK2, lw=1, label="perfect calibration")
ok1 = cnt_b >= 30; ok2 = cnt2 >= 30
ax[0].plot(conf_b[ok1], acc_b[ok1], "o-", color=C1, lw=2, ms=5, label=f"raw confidence (ECE {ece:.3f})")
ax[0].plot(conf2[ok2], acc2[ok2], "s-", color=C2, lw=2, ms=5, label=f"Platt-scaled (ECE {ece2:.3f})")
ax[0].set_xlabel("Stated confidence (bin mean)"); ax[0].set_ylabel("Observed accuracy")
ax[0].set_xlim(0,1); ax[0].set_ylim(0,1); ax[0].set_aspect("equal")
ax[0].set_title("(a) Reliability diagram", loc="left", fontsize=9, color=INK)
ax[0].legend(loc="upper left", fontsize=7.5)
ax[1].bar(bins[:-1]+0.05, cnt_b/N, width=0.09, color=C1, alpha=0.9, label="raw")
ax[1].bar(bins[:-1]+0.05, cnt2/len(ev), width=0.09, color=C2, alpha=0.55, label="Platt-scaled")
ax[1].set_xlabel("Stated confidence"); ax[1].set_ylabel("Fraction of items")
ax[1].set_title("(b) Confidence histogram", loc="left", fontsize=9, color=INK)
ax[1].legend(loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_reliability.png")); plt.close(fig)

# ---------------------------------------------------------------- Figure 3: conformal coverage
rng = np.random.default_rng(3)
alpha = 0.10; n_cal = 200; n_test = 20000; Rr = 3000
k_ord = int(np.ceil((n_cal+1)*(1-alpha)))
K = 5
def gen(m):
    y = rng.integers(0, K, m)
    logits = rng.normal(0, 1.0, (m, K)); logits[np.arange(m), y] += rng.gamma(2.0, 1.2, m)
    p = np.exp(logits); p /= p.sum(1, keepdims=True)
    return y, p
cov = np.empty(Rr); size = np.empty(Rr)
for r in range(Rr):
    yc, pc = gen(n_cal); yt, pt = gen(n_test)
    scores = 1 - pc[np.arange(n_cal), yc]
    qhat = np.sort(scores)[k_ord-1]
    sets = (1 - pt) <= qhat
    cov[r] = sets[np.arange(n_test), yt].mean(); size[r] = sets.sum(1).mean()
beta = stats.beta(k_ord, n_cal+1-k_ord)
results["ch4"].update({"COV_MEAN": f"{cov.mean():.4f}", "COV_SD": f"{cov.std():.4f}",
                       "SET_MEAN": f"{size.mean():.2f}", "UPPER": f"{1-alpha+1/(n_cal+1):.4f}"})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax[0].hist(cov, bins=40, density=True, color=C1, alpha=0.85, label="empirical (3,000 calibration draws)")
xs = np.linspace(0.82, 0.97, 400); ax[0].plot(xs, beta.pdf(xs), color=C2, lw=2, label="Beta(k, n+1−k) theory")
ax[0].axvline(1-alpha, color=INK2, ls="--", lw=1); ax[0].text(1-alpha+0.0015, 1.0, "1−α", color=INK2, fontsize=8)
ax[0].set_xlabel("Test coverage"); ax[0].set_ylabel("Density")
ax[0].set_title("(a) Coverage over calibration draws, n = 200", loc="left", fontsize=9, color=INK)
ax[0].legend(fontsize=7.5, loc="upper left")
yc, pc = gen(n_cal); yt, pt = gen(n_test); scores = 1 - pc[np.arange(n_cal), yc]
levels = np.linspace(0.5, 0.99, 50); sz=[]
for a in 1-levels:
    kk = min(n_cal, int(np.ceil((n_cal+1)*(1-a)))); qq = np.sort(scores)[kk-1]
    sz.append(((1-pt) <= qq).sum(1).mean())
ax[1].plot(levels, sz, color=C1, lw=2)
ax[1].set_xlabel("Target coverage 1−α"); ax[1].set_ylabel("Average set size (of 5 options)")
ax[1].set_title("(b) Price of coverage: set size", loc="left", fontsize=9, color=INK)
ax[1].axvline(0.9, color=INK2, ls="--", lw=1)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_conformal.png")); plt.close(fig)

# ---------------------------------------------------------------- Figure 4: risk-coverage curves
rng = np.random.default_rng(5)
N = 3000
true_p = rng.beta(4, 1.5, N); correct = rng.binomial(1, true_p)
verbal = np.clip(true_p + rng.normal(0, 0.30, N), 0, 1)
consist = np.clip(true_p + rng.normal(0, 0.12, N), 0, 1)
def rc_curve(cf):
    order = np.argsort(-cf); err = 1-correct[order]
    cv = np.arange(1, N+1)/N; risk = np.cumsum(err)/np.arange(1, N+1)
    return cv, risk, np.trapezoid(risk, cv)
START = 30
cv1, r1, aurc1 = rc_curve(verbal); cv2, r2, aurc2 = rc_curve(consist)
rand_risk = 1-correct.mean()
results["ch4"].update({"AURC_V": f"{aurc1:.3f}", "AURC_C": f"{aurc2:.3f}", "BASE_RISK": f"{rand_risk:.3f}",
                       "R70_V": f"{r1[int(0.7*N)-1]:.3f}", "R70_C": f"{r2[int(0.7*N)-1]:.3f}"})
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.plot(cv1[START:], r1[START:], color=C1, lw=2, label=f"verbalized confidence (AURC {aurc1:.3f})")
ax.plot(cv2[START:], r2[START:], color=C2, lw=2, label=f"consistency-based (AURC {aurc2:.3f})")
ax.axhline(rand_risk, color=INK2, ls="--", lw=1); ax.text(0.50, rand_risk+0.006, "answer everything", color=INK2, fontsize=8)
ax.axhline(0.05, color=C3, ls=":", lw=1.2); ax.text(0.02, 0.053, "tolerated risk 5%", color=C3, fontsize=8)
ax.set_xlabel("Coverage (fraction of items answered)"); ax.set_ylabel("Selective risk (error rate on answered items)")
ax.set_xlim(0,1); ax.set_ylim(0, 0.37); ax.legend(loc="upper left", fontsize=7.5)
ax.set_title("Risk–coverage curves for two confidence signals", loc="left", fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_risk_coverage.png")); plt.close(fig)

# ---------------------------------------------------------------- Figure 5 + Tables 6.3/6.4: illustrative study (Ch. 6)
rng = np.random.default_rng(2026)
models = ["Model 1", "Model 2", "Model 3", "Model 4"]
conds = ["API, item-by-item", "Chat, batched"]
n_items = 150
item_eff = rng.normal(0, 0.9, n_items)          # shared item difficulty (rubric)
safety_eff = rng.normal(0, 1.0, n_items)        # separate latent: how "dangerous" an item is
base = {"Model 1": 1.9, "Model 2": 1.6, "Model 3": 1.45, "Model 4": 0.7}
base_unsafe = {"Model 1": -3.9, "Model 2": -2.6, "Model 3": -2.2, "Model 4": -1.5}   # logit of P(unsafe)
cond_shift = {"API, item-by-item": 0.0, "Chat, batched": -0.5}
cond_unsafe = {"API, item-by-item": 0.0, "Chat, batched": 0.6}
score = {}; unsafe6 = {}; severe = {}
for cnd in conds:
    for m in models:
        lg = base[m] + cond_shift[cnd] + item_eff + rng.normal(0, 0.3, n_items)
        p = 1/(1+np.exp(-lg))
        crit = np.clip(p[:, None] + rng.normal(0, 0.15, (n_items, 4)), 0, 1)
        score[(cnd, m)] = crit.mean(1)
        pu = 1/(1+np.exp(-(base_unsafe[m] + cond_unsafe[cnd] + 0.9*safety_eff)))
        u = rng.binomial(1, pu); unsafe6[(cnd, m)] = u
        severe[(cnd, m)] = u * rng.binomial(1, 0.3, n_items)      # 30% of unsafe outputs are severe
tab = {}
for key, s in score.items():
    u, sv = unsafe6[key], severe[key]
    gated = s*(1-u)
    tab[key] = dict(mean=s.mean(), se=se(s), unsafe=u.mean(), n_sev=int(sv.sum()),
                    sev_ub=clopper_upper(int(sv.sum()), n_items), gated=gated.mean(), gated_se=se(gated))
rows6 = []
for m in models:
    cells = []
    for cnd in conds:
        t = tab[(cnd, m)]
        cells.append(f"{t['mean']:.3f} ({t['mean']-1.96*t['se']:.3f}–{t['mean']+1.96*t['se']:.3f}) | {100*t['unsafe']:.1f}% | {t['n_sev']} ({100*t['sev_ub']:.1f}%) | {t['gated']:.3f}")
    rows6.append(f"| {m} | " + " | ".join(cells) + " |")
results["ch6"] = {"CH6_TABLE": "\n".join(rows6)}
# paired comparisons: adjacent models within condition (6) + condition within model (4) = 10, Holm
comps = []
for cnd in conds:
    for a, b in [("Model 1", "Model 2"), ("Model 2", "Model 3"), ("Model 3", "Model 4")]:
        dd = score[(cnd, a)] - score[(cnd, b)]
        comps.append((f"{a} − {b}", cnd.split(",")[0], dd))
for m in models:
    dd = score[(conds[0], m)] - score[(conds[1], m)]
    comps.append((m, "API − Chat", dd))
pvals = np.array([stats.ttest_1samp(dd, 0).pvalue for _, _, dd in comps])
order = np.argsort(pvals); holm = np.empty_like(pvals); mcount = len(pvals)
running = 0.0
for rank, i in enumerate(order):
    adj = min(1.0, (mcount - rank)*pvals[i]); running = max(running, adj); holm[i] = running
rows_pd = []
for (lab_, ctx, dd), p_raw, p_h in zip(comps, pvals, holm):
    rows_pd.append(f"| {lab_} | {ctx} | {dd.mean():+.3f} ({dd.mean()-1.96*se(dd):+.3f}, {dd.mean()+1.96*se(dd):+.3f}) | {p_h:.3f}{'*' if p_h < 0.05 else ''} |")
results["ch6"]["PAIRED_TABLE"] = "\n".join(rows_pd)
results["ch6"]["N_SIG"] = str(int((holm < 0.05).sum()))
# a few named numbers for the prose
imin = int(np.argmin([abs(dd.mean()) for _, _, dd in comps])); dmin = comps[imin][2]
results["ch6"].update({
    "MIN_LABEL": f"{comps[imin][0]} ({comps[imin][1]})",
    "MIN_DIFF": f"{dmin.mean():+.3f} ({dmin.mean()-1.96*se(dmin):+.3f}, {dmin.mean()+1.96*se(dmin):+.3f})",
    "MIN_P": f"{holm[imin]:.3f}",
    "M1_API_SEV": f"{tab[(conds[0], 'Model 1')]['n_sev']}", "M1_API_SEV_UB": f"{100*tab[(conds[0], 'Model 1')]['sev_ub']:.1f}",
    "M1_CHAT_SEV": f"{tab[(conds[1], 'Model 1')]['n_sev']}", "M1_CHAT_SEV_UB": f"{100*tab[(conds[1], 'Model 1')]['sev_ub']:.1f}",
    "M1_API_UNSAFE": f"{100*tab[(conds[0], 'Model 1')]['unsafe']:.1f}", "M1_CHAT_UNSAFE": f"{100*tab[(conds[1], 'Model 1')]['unsafe']:.1f}",
    "M1_API_GATED": f"{tab[(conds[0], 'Model 1')]['gated']:.3f}",
})
# figure: (a) means with CIs, (b) paired differences with 95% CIs, Holm-significant marked
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1, 1.15]})
ypos = np.arange(len(models))
for j, (cnd, col, mk) in enumerate(zip(conds, [C1, C2], ["o", "s"])):
    vals = [tab[(cnd, m)] for m in models]
    ax[0].errorbar([v["mean"] for v in vals], ypos + (0.15 if j else -0.15), xerr=[1.96*v["se"] for v in vals],
                   fmt=mk, color=col, ecolor=col, capsize=3, ms=6, lw=1.5, label=cnd)
ax[0].set_yticks(ypos); ax[0].set_yticklabels(models); ax[0].invert_yaxis()
ax[0].set_xlabel("Mean rubric score with 95% CI"); ax[0].set_xlim(0.4, 1.0)
ax[0].legend(loc="upper left", fontsize=7.5); ax[0].set_title("(a) Scores by model and condition", loc="left", fontsize=9, color=INK)
labels = [f"{l} ({c})" if c != "API − Chat" else f"{l}: API − Chat" for l, c, _ in comps]
means = np.array([dd.mean() for _, _, dd in comps]); halfs = np.array([1.96*se(dd) for _, _, dd in comps])
cols = [C1 if c == "API" else (C2 if c == "Chat" else C3) for _, c, _ in comps]
yy = np.arange(len(comps))
for i in range(len(comps)):
    ax[1].errorbar(means[i], yy[i], xerr=halfs[i], fmt="o", color=cols[i], ecolor=cols[i], capsize=3, ms=5, lw=1.5)
    if holm[i] < 0.05: ax[1].text(means[i]+halfs[i]+0.01, yy[i], "*", color=INK, va="center", fontsize=10)
ax[1].axvline(0, color=INK2, ls="--", lw=1)
ax[1].set_yticks(yy); ax[1].set_yticklabels(labels, fontsize=7); ax[1].invert_yaxis()
ax[1].set_xlabel("Paired difference, 95% CI (* Holm p < 0.05)")
ax[1].set_title("(b) Pre-specified paired comparisons", loc="left", fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_illustrative.png")); plt.close(fig)


# ---------------------------------------------------------------- Figure 6 + Table 5.1: severity and perturbation (Ch. 5)
rng = np.random.default_rng(55)
n5 = 200
sev_models = ["Model X", "Model Y", "Model Z"]
err_rate = {"Model X": 0.10, "Model Y": 0.15, "Model Z": 0.25}
# severity distribution among errors (0 none, 1 minor, 2 moderate, 3 severe)
sev_dist = {"Model X": [0.30, 0.30, 0.20, 0.20], "Model Y": [0.50, 0.30, 0.15, 0.05], "Model Z": [0.60, 0.30, 0.08, 0.02]}
cost_w = np.array([0, 1, 5, 25])     # illustrative harm weights per severity level
rows5 = []; sev_counts = {}
for m in sev_models:
    e = rng.binomial(1, err_rate[m], n5)
    v = np.where(e == 1, rng.choice(4, n5, p=sev_dist[m]), 0)
    counts = np.array([((e == 1) & (v == k)).sum() for k in range(4)]); sev_counts[m] = counts
    n_err = int(e.sum()); n_sev = int(counts[3])
    exp_sev = v.mean(); exp_harm = cost_w[v].mean()
    lo_e, hi_e = wilson(n_err, n5)
    rows5.append(f"| {m} | {n_err/n5:.3f} ({lo_e:.3f}–{hi_e:.3f}) | {counts[0]}/{counts[1]}/{counts[2]}/{counts[3]} | {exp_sev:.3f} | {n_sev} ({100*clopper_upper(n_sev, n5):.1f}%) | {exp_harm:.2f} |")
results["ch5"] = {"SEV_TABLE": "\n".join(rows5)}
# perturbation: paraphrase each item; two models with the same mean change but different flip rates
pert = {}
for m, p_corr, flip in [("Model X", 0.85, 0.06), ("Model Z", 0.75, 0.22)]:
    base_c = rng.binomial(1, p_corr, n5)
    do_flip = rng.binomial(1, flip, n5)
    pert_c = np.where(do_flip == 1, 1 - base_c, base_c)
    d = pert_c - base_c
    lo_f, hi_f = wilson(int(do_flip.sum()), n5)
    pert[m] = dict(flip=do_flip.mean(), flip_ci=(lo_f, hi_f), change=d.mean(), change_se=se(d.astype(float)), base=base_c.mean(), after=pert_c.mean())
results["ch5"].update({
    "FLIP_X": f"{100*pert['Model X']['flip']:.1f}% ({100*pert['Model X']['flip_ci'][0]:.1f}–{100*pert['Model X']['flip_ci'][1]:.1f}%)",
    "FLIP_Z": f"{100*pert['Model Z']['flip']:.1f}% ({100*pert['Model Z']['flip_ci'][0]:.1f}–{100*pert['Model Z']['flip_ci'][1]:.1f}%)",
    "CHG_X": f"{pert['Model X']['change']:+.3f} ({pert['Model X']['change']-1.96*pert['Model X']['change_se']:+.3f}, {pert['Model X']['change']+1.96*pert['Model X']['change_se']:+.3f})",
    "CHG_Z": f"{pert['Model Z']['change']:+.3f} ({pert['Model Z']['change']-1.96*pert['Model Z']['change_se']:+.3f}, {pert['Model Z']['change']+1.96*pert['Model Z']['change_se']:+.3f})",
    "ACC_X": f"{pert['Model X']['base']:.3f}", "ACC_Z": f"{pert['Model Z']['base']:.3f}",
    "ERR_X": f"{err_rate['Model X']:.2f}", "ERR_Z": f"{err_rate['Model Z']:.2f}",
})
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.2, 1]})
ramp = ["#c9ddf5", "#7fb0e8", "#2a78d6", "#123f78"]       # one hue, light -> dark (ordinal severity)
left = np.zeros(len(sev_models))
for k in range(4):
    vals = np.array([sev_counts[m][k] for m in sev_models])/n5*100
    ax[0].barh(sev_models, vals, left=left, color=ramp[k], edgecolor="white", linewidth=1.5, label=["no harm", "minor", "moderate", "severe"][k])
    left += vals
ax[0].invert_yaxis(); ax[0].set_xlabel("Errors as % of items, by severity"); ax[0].legend(fontsize=7.5, loc="upper right", ncol=2)
ax[0].set_title("(a) Error rate and severity (200 items)", loc="left", fontsize=9, color=INK)
labels = ["Model X", "Model Z"]
ax[1].errorbar([100*pert[m]["flip"] for m in labels], [0, 1], xerr=[[100*(pert[m]["flip"]-pert[m]["flip_ci"][0]) for m in labels], [100*(pert[m]["flip_ci"][1]-pert[m]["flip"]) for m in labels]],
               fmt="o", color=C2, ecolor=C2, capsize=3, ms=6, lw=1.5, label="flip rate (% of items)")
ax[1].errorbar([100*pert[m]["change"] for m in labels], [0.25, 1.25], xerr=[100*1.96*pert[m]["change_se"] for m in labels],
               fmt="s", color=C1, ecolor=C1, capsize=3, ms=6, lw=1.5, label="mean change in accuracy (points)")
ax[1].axvline(0, color=INK2, ls="--", lw=1); ax[1].set_yticks([0.1, 1.1]); ax[1].set_yticklabels(labels); ax[1].invert_yaxis()
ax[1].set_xlabel("Under paraphrase perturbation"); ax[1].legend(fontsize=7, loc="upper right"); ax[1].set_xlim(-12, 36)
ax[1].set_title("(b) Robustness: flips vs. mean change", loc="left", fontsize=9, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_severity_robustness.png")); plt.close(fig)

with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(json.dumps(results, indent=2, ensure_ascii=True))

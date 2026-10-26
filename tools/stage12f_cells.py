MD_12F = '''### 12f · Stability first: keep the old classes, then learn as much "zero" as possible

The selection rule changes here: a setting is **acceptable** only if old-class accuracy on validation falls by at most `max_old_drop_pts` (2 points) below the base network's; among acceptable settings, the one with the highest validation "zero" accuracy wins. Four ways to protect the old classes, each a dial from "learn freely" to "barely move":

- **Calibration** (after training, no retraining): a bias added to the "zero" output, chosen on validation as the highest "zero" accuracy that keeps the old classes within the limit. It removes stealing; it cannot undo drift.
- **Distillation** (Learning without Forgetting): the 12 old logits are kept close to the base network's on every clip of the batch. It needs no labels, so it uses exactly the data the no-replay runs see. Strength λ swept, with no mask and with the best mask from 12c.
- **Soft freezing**: a penalty on how far each neuron's incoming weights move, weighted equally (plain L2), by the neuron's Fisher information, or by its best-blend MI + Fisher score. Strength λ swept, no mask.
- **Freezing 80–95%** of the hidden neurons (best blend and random), to complete the 12c curve.

Every run here keeps its validation and test logits, so each is reported as trained and calibrated. The rule is also applied to every earlier run, as trained.

**Check:** frozen weights never moved; calibration with bias 0 reproduces the trained scores; at least one setting meets the limit.'''

CODE_12F = r'''ST = CFG["stability"]
TOL = ST["max_old_drop_pts"] / 100.0
BASE_VAL_OLD = float(BASE_OLD_VAL.mean())


def logit_scores(Z, Y, base_old):
    s = summarize(Z, Y, base_old)
    s["hmean"] = hmean(s["old_mean_acc"], s["new_word_acc"])
    return {k: s[k] for k in KEEP if k in s}


def old_new(Z, Y, b):
    """Old-class (mean per class) and new-word accuracy with bias b added to the new output."""
    Zb = Z.clone()
    Zb[:, NEW] += b
    pc, _ = class_scores(decide(Zb), Y, 13)
    return float(pc[:NEW].mean()), float(pc[NEW])


def calibrate(L):
    """Bias on the new output, chosen on validation: the highest new-word accuracy whose old-class accuracy stays within
    TOL of the base network; if no bias gets there, the bias with the highest old-class accuracy."""
    Zv, Yv, Zt, Yt = L["val_Z"], L["val_Y"], L["test_Z"], L["test_Y"]
    g = (Zv[:, NEW] - Zv[:, :NEW].max(1).values).float()
    qs = torch.quantile(g, torch.linspace(0, 1, 201))
    biases = sorted({0.0} | {float(-q + d) for q in qs for d in (-1e-4, 1e-4)})
    rows = [(b, *old_new(Zv, Yv, b)) for b in biases]
    ok = [r for r in rows if r[1] >= BASE_VAL_OLD - TOL]
    b = max(ok, key=lambda r: (r[2], r[1]))[0] if ok else max(rows, key=lambda r: (r[1], r[2]))[0]
    Zv2, Zt2 = Zv.clone(), Zt.clone()
    Zv2[:, NEW] += b
    Zt2[:, NEW] += b
    return {"bias": b, "feasible": bool(ok), "val": logit_scores(Zv2, Yv, BASE_OLD_VAL), "test": logit_scores(Zt2, Yt, BASE_OLD),
            "val_curve": [[r[0], r[1], r[2]] for r in rows[::4]]}


def stab_run(name, frozen, stage="12f_stability", seed=0, **kw):
    """One run that keeps its logits and is also scored after calibration."""
    def fn():
        r = train_cl(frozen, keep_logits=True, tag=name, seed=seed, **kw)
        L = r.pop("logits")
        save_pt(L, stage_dir(stage) / "runs" / f"{name}_logits.pt")
        r["calibrated"] = calibrate(L)
        return r
    return run_cached(stage, name, fn)


def importance(kind):
    """Per-neuron importance for soft freezing, scaled to mean 1 over all hidden neurons."""
    if kind == "uniform":
        v = torch.ones(N_H, dtype=torch.float64)
    elif kind == "fisher":
        v = FISH["naive"].double().mean(0)
    elif kind == "blend":
        wl = BEST_BLEND_MODE == "within_layer"
        v = (BEST_ALPHA * rank_scores(MI["per_class"], wl) + (1 - BEST_ALPHA) * rank_scores(FISH["naive"], wl)).max(0).values
    else:
        raise ValueError(kind)
    v = v / v.mean()
    return [v[MODEL.offsets[l]:MODEL.offsets[l + 1]].float().to(DEVICE) for l in range(L_H)]


BEST_SET = budget_set(BEST_METHOD, BEST_B)
STAB_CFGS = {"ref_readout_only": (frozen_set("all"), {}), "ref_no_mask": (set(), {}), "ref_mi_mask": (frozen_set("mi"), {}),
             "ref_best_blend": (frozen_set("combo:" + BEST_BLEND), {}), "ref_best_mask": (BEST_SET, {}),
             "ref_replay_25": (set(), {"replay_frac": 0.25})}           # name: (frozen neurons, train_cl options)
for mk, fz in [("none", set()), ("best", BEST_SET)]:
    for lam in ST["lwf_lambdas"]:
        STAB_CFGS[f"lwf_{mk}_{lam:g}"] = (fz, {"lwf": lam})
for kind in ST["soft_importance"]:
    imp = importance(kind)
    for lam in ST["soft_lambdas"]:
        STAB_CFGS[f"soft_{kind}_{lam:g}"] = (set(), {"soft": {"weights": imp, "lam": lam}})
for m in ST["extreme_methods"]:
    for frac in ST["extreme_budgets"]:
        STAB_CFGS[f"freeze_{m}_b{int(round(100 * frac)):02d}"] = (budget_set(m, max(1, int(round(frac * N_H)))), {})
STAB_RUNS = {name: stab_run(name, fz, **kw) for name, (fz, kw) in STAB_CFGS.items()}
'''

CHK_12F = r'''# ---- CHECK 12f: the stability-first selection
frozen_ok("12f", STAB_RUNS)
L0 = load_pt(stage_dir("12f_stability") / "runs" / "ref_no_mask_logits.pt")
o0, n0 = old_new(L0["test_Z"], L0["test_Y"], 0.0)
check("12f calibration with bias 0 reproduces the trained test scores",
      abs(o0 - STAB_RUNS["ref_no_mask"]["test"]["old_mean_acc"]) < 1e-6 and abs(n0 - STAB_RUNS["ref_no_mask"]["test"]["new_word_acc"]) < 1e-6,
      f"old {o0:.4f}, new {n0:.4f}")
print(f"base network old-class accuracy: {BASE_VAL_OLD:.3f} (validation), {float(BASE_OLD.mean()):.3f} (test); "
      f"limit: at most {ST['max_old_drop_pts']:g} points lower on validation\n")
print(f"{'run':<26}{'old':>8}{'zero':>8}{'all 13':>9}{'drift':>8}   |  calibrated: {'bias':>7}{'ok':>4}{'old':>8}{'zero':>8}{'all 13':>9}   (test)")
for n, r in STAB_RUNS.items():
    t, c = r["test"], r["calibrated"]
    print(f"{n:<26}{t['old_mean_acc']:>8.3f}{t['new_word_acc']:>8.3f}{t['acc_13']:>9.3f}{t['drift_forgetting_mean_pts']:>8.1f}   |  "
          f"{'':>12}{c['bias']:>7.2f}{'yes' if c['feasible'] else 'no':>4}{c['test']['old_mean_acc']:>8.3f}"
          f"{c['test']['new_word_acc']:>8.3f}{c['test']['acc_13']:>9.3f}")

CANDIDATES = {}
for grp, runs in [("12b", ARM_RUNS), ("12c", BUDGET_RUNS), ("12d", TRADE_RUNS), ("12f", STAB_RUNS)]:
    for n, r in runs.items():
        CANDIDATES[f"{grp}/{n}"] = {"val": r["val"], "test": r["test"]}
for n, r in STAB_RUNS.items():
    CANDIDATES[f"12f/{n} (calibrated)"] = {"val": r["calibrated"]["val"], "test": r["calibrated"]["test"]}
CANDIDATES["12b/slda_all"] = {"val": SLDA_RUN["val"], "test": SLDA_RUN["test"]}
feasible = {n: c for n, c in CANDIDATES.items() if c["val"]["old_mean_acc"] >= BASE_VAL_OLD - TOL}
ranked = sorted(feasible, key=lambda n: (-feasible[n]["val"]["new_word_acc"], -feasible[n]["val"]["old_mean_acc"]))
print(f"\n{len(feasible)} of {len(CANDIDATES)} settings keep validation old-class accuracy within {ST['max_old_drop_pts']:g} points; "
      f"best by validation '{NEW_WORD}' accuracy:")
for n in ranked[:15]:
    t = feasible[n]["test"]
    print(f"  {n:<44} test: old {t['old_mean_acc']:.3f} (drop {100 * (float(BASE_OLD.mean()) - t['old_mean_acc']):+.1f} pts), "
          f"'{NEW_WORD}' {t['new_word_acc']:.3f}, all 13 {t['acc_13']:.3f}")
STAB_BEST = ranked[0] if ranked else None
check("12f at least one setting meets the stability limit on validation", bool(ranked), STAB_BEST or "none")
STAB_SUMMARY = {"tolerance_pts": ST["max_old_drop_pts"], "base_val_old": BASE_VAL_OLD, "base_test_old": float(BASE_OLD.mean()),
                "best": STAB_BEST, "best_test": feasible[STAB_BEST]["test"] if STAB_BEST else None,
                "feasible_ranked": [{"name": n, "val": feasible[n]["val"], "test": feasible[n]["test"]} for n in ranked[:30]],
                "runs": {n: {"test": r["test"], "val": r["val"], "settings": r.get("settings"), "weight_change": r.get("weight_change"),
                             "calibrated": r["calibrated"]} for n, r in STAB_RUNS.items()}}
if not is_done("12f_stability"):
    mark_done("12f_stability")
log(f"12f_stability: best setting within the limit = {STAB_BEST}")
if HAVE_PLT:
    plt.figure(figsize=(7, 5))
    for n, r in STAB_RUNS.items():
        t, c = r["test"], r["calibrated"]["test"]
        plt.plot([t["old_mean_acc"], c["old_mean_acc"]], [t["new_word_acc"], c["new_word_acc"]], c="lightgrey", lw=0.8)
        plt.scatter(t["old_mean_acc"], t["new_word_acc"], s=14, c="grey")
        plt.scatter(c["old_mean_acc"], c["new_word_acc"], s=14, c="tab:blue")
    plt.axvline(float(BASE_OLD.mean()), c="k", lw=0.8, ls=":")
    plt.axvline(float(BASE_OLD.mean()) - TOL, c="tab:red", lw=0.8, ls="--")
    plt.xlabel("old-class accuracy (test)")
    plt.ylabel(f"'{NEW_WORD}' accuracy (test)")
    plt.title("12f: as trained (grey) and calibrated (blue); dashed = stability limit")
    savefig("12f_stability")
'''

MD_12G = '''### 12g · Seeds for the stability-first winners

The top settings of the 12f ranking (the `top_k` best runs that meet the limit on validation, as trained or calibrated), plus readout-only and 25% replay as references, repeated with the seeds in `CFG["stability_seeds"]["seeds"]`. Seed 0 is the 12f run itself. Each run is calibrated on its own validation logits, exactly as in 12f. Reported as mean ± SD, as trained and calibrated, with the test drop in old-class accuracy against the base network.'''

CODE_12G = r'''SS = CFG["stability_seeds"]
TOP12F = []
for n in ranked:
    if n.startswith("12f/"):
        b = n[len("12f/"):].replace(" (calibrated)", "")
        if b not in TOP12F:
            TOP12F.append(b)
    if len(TOP12F) >= SS["top_k"]:
        break
for ref in SS["references"]:
    if ref not in TOP12F:
        TOP12F.append(ref)
log(f"12g: seeds {SS['seeds']} for {TOP12F}")
SEED12G = {}
for name in TOP12F:
    fz, kw = STAB_CFGS[name]
    SEED12G[name] = {0: STAB_RUNS[name]}
    for sd in SS["seeds"]:
        SEED12G[name][sd] = stab_run(f"{name}_s{sd}", fz, stage="12g_stability_seeds", seed=sd, **kw)
'''

CHK_12G = r'''# ---- CHECK 12g: mean +- SD over seeds, as trained and calibrated
frozen_ok("12g", {f"{n}/{s}": r for n, runs in SEED12G.items() for s, r in runs.items() if s != 0})
if SS["seeds"]:
    n0 = TOP12F[0]
    check("12g different seeds give different runs",
          len({round(r["test"]["old_mean_acc"], 6) for r in SEED12G[n0].values()}) > 1, n0)


def ms(vals):
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals))}


base_old_test = float(BASE_OLD.mean())
SEED12G_SUMMARY = {}
for name, runs in SEED12G.items():
    rs = list(runs.values())
    SEED12G_SUMMARY[name] = {
        "seeds": sorted(runs),
        "trained": {m: ms([r["test"][m] for r in rs]) for m in ("old_mean_acc", "new_word_acc", "acc_13", "drift_forgetting_mean_pts")},
        "calibrated": {m: ms([r["calibrated"]["test"][m] for r in rs]) for m in ("old_mean_acc", "new_word_acc", "acc_13")},
        "old_drop_pts_calibrated": ms([100 * (base_old_test - r["calibrated"]["test"]["old_mean_acc"]) for r in rs]),
        "feasible_on_val": int(sum(bool(r["calibrated"]["feasible"]) for r in rs)), "n": len(rs)}
print(f"base network old-class accuracy (test) {base_old_test:.3f}\n")
print(f"{'setting':<22}{'n':>3}   {'as trained: old / zero / all 13':<34}   {'calibrated: old / zero / all 13':<34}{'old drop (pts)':>16}")
for name, s in SEED12G_SUMMARY.items():
    tr, ca = s["trained"], s["calibrated"]
    f = lambda d: f"{d['mean']:.3f}±{d['sd']:.3f}"
    print(f"{name:<22}{s['n']:>3}   {f(tr['old_mean_acc'])} {f(tr['new_word_acc'])} {f(tr['acc_13'])}   "
          f"{f(ca['old_mean_acc'])} {f(ca['new_word_acc'])} {f(ca['acc_13'])}"
          f"{s['old_drop_pts_calibrated']['mean']:>9.2f} ± {s['old_drop_pts_calibrated']['sd']:.2f}")
if not is_done("12g_stability_seeds"):
    mark_done("12g_stability_seeds")
'''

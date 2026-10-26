"""Add two ablations to snn_gscv2_pipeline.ipynb:
  - stage 11b + stage-12 arms: neuron groups from a combined MI + Fisher score (BPTT base and BPTT continual)
  - stage-12 arms: forward-forward learning of the hidden layers during continual learning (BPTT base)
plus a layer0_only control arm."""
import json, sys, uuid
from pathlib import Path

path = Path(sys.argv[1])
nb = json.loads(path.read_text(encoding="utf-8"))
cells = nb["cells"]


def src(i):
    return "".join(cells[i]["source"])


def set_src(i, s):
    cells[i]["source"] = s.splitlines(keepends=True)


def replace_once(i, old, new):
    s = src(i)
    assert s.count(old) == 1, f"cell {i}: anchor found {s.count(old)} times: {old[:80]!r}"
    set_src(i, s.replace(old, new))


def new_cell(kind, text):
    c = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


# locate cells by content, not by index
def find(prefix):
    hits = [i for i, c in enumerate(cells) if "".join(c["source"]).lstrip().startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


C_INTRO = 0
C_CFG = find("# ---------------------------------------------------------------- what to run, and where")
C_HELP = find("# ---------------------------------------------------------------- small helpers used by every stage")
C_CHK11 = find("# ---- CHECK 11: the ablation tables")
C_MD12 = find("## 12 · Continual learning")
C_CL = find("NB_INIT, BS_CL = ")
C_ARMS = find("# ---- the arms")
C_RUN = find("# ---- run every arm")
C_SUM = find("def compact_arm(r):")

# ---------------------------------------------------------------- intro table
replace_once(C_INTRO, "| 11 | Ablation and sufficiency on the held-out test split | `11_ablation/` |\n",
             "| 11 | Ablation and sufficiency on the held-out test split | `11_ablation/` |\n"
             "| 11b | Groups from a combined MI + Fisher score, and their ablation | `11b_combo/` |\n")
replace_once(C_INTRO, '| 12 | Continual learning: add "zero" (13 arms) and check hypotheses P1–P6 | `12_continual/` |',
             '| 12 | Continual learning: add "zero" (13 arms + combined-score, layer-0 and forward-forward arms) and check hypotheses P1–P6 | `12_continual/` |')

# ---------------------------------------------------------------- config
replace_once(C_CFG, '''                 "ksweep": [20, 40, 80, 160, 320], "save_full_fisher": True},
''', '''                 "ksweep": [20, 40, 80, 160, 320], "save_full_fisher": True},
    "combo": {"alphas": [0.0, 0.25, 0.5, 0.75, 1.0],      # group score = alpha * MI rank + (1 - alpha) * Fisher rank
              "rank_modes": ["global", "within_layer"]},   # ranks over all hidden neurons, or within each layer
''')
replace_once(C_CFG, '''                    "mi_mask_replay", "no_mask_replay", "one_way_new_neurons"]},
''', '''                    "mi_mask_replay", "no_mask_replay", "one_way_new_neurons",
                    "layer0_only", "combo", "ff_contrast", "ff_contrast_layer0_frozen", "ff_hybrid"],
           "ff": {"kappa": 4.0, "hybrid_block": 10}},     # forward-forward: goodness slope, hybrid-negative block (steps)
''')

# ---------------------------------------------------------------- stage bookkeeping
replace_once(C_HELP, '''          "08_fisher", "09_mi", "10_groups", "11_ablation", "12_continual"]''',
             '''          "08_fisher", "09_mi", "10_groups", "11_ablation", "11b_combo", "12_continual"]''')
replace_once(C_HELP, '''"08_fisher": ["analysis"], "12_continual": ["cl"]}''',
             '''"08_fisher": ["analysis"], "11b_combo": ["combo"],
             "12_continual": ["cl"]}''')

# ---------------------------------------------------------------- stage 11b (new cells after CHECK 11)
MD_11B = '''## 11b · Groups from a combined MI + Fisher score (held-out test split)

Stage 10 picks each class's group with one measure. Here the two are blended. For class c and neuron i:

score = α · rank(MIᶜᵢ) + (1 − α) · rank(Fᶜᵢᵢ),

where each rank is a percentile in [0, 1], and α runs over `CFG["combo"]["alphas"]` (α = 1 is pure MI, α = 0 pure naive Fisher). Each class keeps its top-K neurons by this score. The ranks come in two flavours:

- **global:** ranked over all hidden neurons. Fisher's depth bias stays in, so a low α pulls the group into layer 0. With these ranks, α = 1 and α = 0 reproduce the stage-10 MI and Fisher groups (a check below).
- **within_layer:** each neuron is ranked only against its own layer. The depth bias is removed, so α = 0 is a "Fisher within layer" selector and the groups spread over every layer.

Every group set gets the stage-11 per-class ablation (same test clips, same baseline), and stage 12 freezes the union of each set's groups in its own arm (`combo_<global|layer>_a<100·α>`). The base network and every continual arm here are trained with BPTT.

**Check:** with global ranks, α = 1 and α = 0 match the MI and Fisher groups; every group has K valid ids. Then the table: union size and layers, overlap with the MI and Fisher groups, and specificity per α.

Saved: `11b_combo/combo.json`.'''

CODE_11B = '''d11b = stage_dir("11b_combo")


def rank_scores(score, within_layer):
    """Percentile rank per class (0 = lowest, 1 = highest), over all hidden neurons or within each layer. [C, N]"""
    out = torch.zeros(score.shape, dtype=torch.float64)
    parts = layer_sl if within_layer else [slice(0, score.shape[1])]
    for c in range(score.shape[0]):
        for s in parts:
            r = avg_ranks(score[c, s])
            out[c, s] = r / max(len(r) - 1, 1)
    return out


def combo_name(mode, alpha):
    return f"combo_{'layer' if mode == 'within_layer' else 'global'}_a{int(round(100 * alpha)):03d}"


def mean_overlap(groups, ref):
    return float(np.mean([len(set(groups[c]) & set(ref[c])) / K for c in range(len(ref))]))


# ties (e.g. the best neuron of every layer at rank 1 within layers) are broken by a tiny global-rank term
TIEBREAK = 1e-6 * (rank_scores(MI["per_class"], False) + rank_scores(FISH["naive"], False))
COMBO_GROUPS, COMBO_META = {}, {}
for mode in CFG["combo"]["rank_modes"]:
    r_mi = rank_scores(MI["per_class"], mode == "within_layer")
    r_f = rank_scores(FISH["naive"], mode == "within_layer")
    for a in CFG["combo"]["alphas"]:
        name = combo_name(mode, a)
        COMBO_GROUPS[name] = topk_per_class(a * r_mi + (1 - a) * r_f + TIEBREAK, K)
        COMBO_META[name] = {"mode": mode, "alpha": float(a)}

if is_done("11b_combo"):
    COMBO = load_json(d11b / "combo.json")
    log("11b_combo: loaded")
else:
    t0 = time.time()
    base_pc = ABL["baseline"]["per_class"]
    COMBO = {}
    for name, groups in COMBO_GROUPS.items():
        rows = []
        for c in range(12):
            ev = ablate(groups[c])
            tgt, oth, spec = drops_vs(base_pc, ev, c)
            rows.append({"class": CLASSES_12[c], "target_drop": tgt, "others_drop": oth, "specificity": spec,
                         "no_decision": ev["n_tied"]})
        union = set().union(*[set(g) for g in groups.values()])
        COMBO[name] = {**COMBO_META[name], "specificity": float(np.mean([r["specificity"] for r in rows])),
                       "target_drop": float(np.mean([r["target_drop"] for r in rows])),
                       "others_drop": float(np.mean([r["others_drop"] for r in rows])),
                       "union_size": len(union), "union_per_layer": [sum(MODEL.layer_of(x) == l for x in union) for l in range(L_H)],
                       "overlap_mi": mean_overlap(groups, GROUPS["mi"]), "overlap_fisher": mean_overlap(groups, GROUPS["fisher"]),
                       "stats": group_stats(groups), "rows": rows}
        log(f"  11b: {name} ablated ({time.time() - t0:.0f} s)")
    save_json(COMBO, d11b / "combo.json")
    mark_done("11b_combo", seconds=round(time.time() - t0))
    log(f"11b_combo: done in {time.time() - t0:.0f} s")
'''

CHK_11B = '''# ---- CHECK 11b: the combined-score groups
check("11b every combined group holds K valid neuron ids",
      all(len(g) == K and all(0 <= x < N_H for x in g) for gs in COMBO_GROUPS.values() for g in gs.values()))
if "global" in CFG["combo"]["rank_modes"] and {0.0, 1.0} <= set(CFG["combo"]["alphas"]):
    o_mi = mean_overlap(COMBO_GROUPS[combo_name("global", 1.0)], GROUPS["mi"])
    o_f = mean_overlap(COMBO_GROUPS[combo_name("global", 0.0)], GROUPS["fisher"])
    check("11b global ranks: alpha = 1 gives the MI groups, alpha = 0 the Fisher groups", o_mi >= 0.95 and o_f >= 0.95,
          f"overlap {o_mi:.0%} with MI, {o_f:.0%} with Fisher (ties in the scores can move a neuron or two)")
print(f"{'group set':<20}{'alpha':>6}{'union':>7}{'union per layer':>17}{'vs MI':>8}{'vs Fisher':>11}"
      f"{'own drop':>10}{'others':>8}{'specificity':>13}   (drops in points)")
for name, r in COMBO.items():
    print(f"{name:<20}{r['alpha']:>6.2f}{r['union_size']:>7}{str(r['union_per_layer']):>17}{r['overlap_mi']:>8.0%}"
          f"{r['overlap_fisher']:>11.0%}{100 * r['target_drop']:>10.1f}{100 * r['others_drop']:>8.1f}{r['specificity']:>+13.3f}")
print(f"stage 11 for reference: specificity MI {SPEC['mi']:+.3f}, Fisher {SPEC['fisher']:+.3f}, Omega {SPEC['omega']:+.3f}")

if HAVE_PLT:
    plt.figure(figsize=(6, 3.5))
    for mode in CFG["combo"]["rank_modes"]:
        rs = sorted((r for r in COMBO.values() if r["mode"] == mode), key=lambda r: r["alpha"])
        plt.plot([r["alpha"] for r in rs], [r["specificity"] for r in rs], marker="o", label=f"{mode} ranks")
    plt.xlabel("alpha (0 = Fisher only, 1 = MI only)")
    plt.ylabel("ablation specificity")
    plt.legend()
    plt.title("combined MI + Fisher groups (test split)")
    savefig("11b_combo_specificity")
'''

cells[C_CHK11 + 1:C_CHK11 + 1] = [new_cell("markdown", MD_11B), new_cell("code", CODE_11B), new_cell("code", CHK_11B)]
C_MD12, C_CL, C_ARMS, C_RUN, C_SUM = (i + 3 for i in (C_MD12, C_CL, C_ARMS, C_RUN, C_SUM))

# ---------------------------------------------------------------- stage 12 description
replace_once(C_MD12, '''| `one_way_new_neurons` | 64 new LIF neurons beside the last hidden layer + the new row; they read the frozen layer below, and only the new output reads them | gradient, no replay | yes (new neurons only) |
''', '''| `one_way_new_neurons` | 64 new LIF neurons beside the last hidden layer + the new row; they read the frozen layer below, and only the new output reads them | gradient, no replay | yes (new neurons only) |
| `layer0_only` | all but layer 0 (control: is it layer 0 or the chosen neurons that protects?) | gradient, no replay | yes |
| `combo_<global\\|layer>_a<α>` | all but the union of the stage-11b groups for that rank mode and α | gradient, no replay | yes |
| `ff_contrast` | every hidden weight + the new row | **forward-forward**: each layer raises its goodness on "zero" clips and lowers it on old clips (no old labels); new row by the local rule | **no** |
| `ff_contrast_layer0_frozen` | as `ff_contrast`, layer 0 frozen | as `ff_contrast` | **no** |
| `ff_hybrid` | every hidden weight + the new row | **forward-forward**, label-free: goodness up on every real clip, down on "hybrid" clips spliced from two clips in blocks of 10 steps; new row by the local rule | **no** |
''')
replace_once(C_MD12, '''- **No decision:**''', '''- **Forward-forward arms** (Hinton 2022, adapted to LIF neurons). The base network is the same BPTT network as everywhere else. A layer's goodness is the mean over its neurons of the squared firing rate. Each layer has its own loss, softplus(−z) on positive clips + softplus(z) on negative clips with z = κ (goodness − θ) / θ, and θ is set per layer to the base network's mean goodness on positives and negatives, so training starts balanced. Every layer reads its input detached, and its membrane from the previous step detached, so a spike's gradient reaches only the weights into that neuron at that step: no backprop across layers and none through time. A check confirms both. The old readout rows stay frozen; the new row learns by the same local rule as `local_rule`.
- **No decision:**''')

# ---------------------------------------------------------------- frozen sets
replace_once(C_CL, '''    if kind == "mi+fisher":
        return mi | fisher_top
    raise ValueError(kind)''', '''    if kind == "mi+fisher":
        return mi | fisher_top
    if kind == "layer0":
        return layer0
    if kind.startswith("combo:"):                   # the union of one stage-11b group set
        return set().union(*[set(g) for g in COMBO_GROUPS[kind.split(":", 1)[1]].values()])
    raise ValueError(kind)''')

# ---------------------------------------------------------------- arms
replace_once(C_ARMS, '''    "no_mask_replay": ("none", True),
}
''', '''    "no_mask_replay": ("none", True),
    "layer0_only": ("layer0", False),
}
for _name in COMBO_GROUPS:                           # one arm per stage-11b group set
    BPTT_ARMS[_name] = ("combo:" + _name, False)
''')

FF_CODE = '''

# ---- forward-forward arms: the hidden layers learn without backprop, the new row by the local rule
FF_ARMS = {                         # name: (negative data, frozen hidden neurons)
    "ff_contrast": ("old", "none"),
    "ff_contrast_layer0_frozen": ("old", "layer0"),
    "ff_hybrid": ("hybrid", "none"),
}


def ff_counts(model, x):
    """The normal forward pass, except that every layer reads its input DETACHED and its own membrane from the
    previous step DETACHED. Spike values are unchanged; a spike's gradient now reaches only the weights into that
    neuron, at that step (no backprop across layers or through time). Returns per-layer spike counts [B, N_l]."""
    T, B, _ = x.shape
    mems = [x.new_zeros(B, n) for n in model.hidden]
    counts = [x.new_zeros(B, n) for n in model.hidden]
    for t in range(T):
        h = x[t]
        for l in range(len(model.hidden)):
            s, mems[l] = lif_step(mems[l].detach(), model.fc[l](h.detach()), model.beta, model.thr)
            counts[l] = counts[l] + s
            h = s
    return counts


def ff_goodness(counts):
    """Goodness of each layer for each clip: mean over its neurons of the squared firing rate. [L, B]"""
    return torch.stack([(c / T_STEPS).pow(2).mean(1) for c in counts])


def hybrid_negatives(x, seed):
    """Negative data in time (Hinton's hybrid images): each clip's spike train is spliced with another clip's in
    random blocks of `hybrid_block` steps, so local structure is real but the whole is no word."""
    T, B, _ = x.shape
    blk = CFG["cl"]["ff"]["hybrid_block"]
    g = torch.Generator().manual_seed(seed)
    own = (torch.rand(math.ceil(T / blk), B, generator=g) < 0.5).repeat_interleave(blk, 0)[:T].to(x.device)
    shift = int(torch.randint(1, max(B, 2), (1,), generator=g))
    partner = (torch.arange(B, device=x.device) + shift) % B
    return torch.where(own[:, :, None], x, x[:, partner])


def ff_split(x, y, neg, seed):
    """Positive and negative inputs of one batch. 'old': positives = new-word clips, negatives = old clips (their
    labels are never used). 'hybrid': positives = every real clip, negatives = spliced clips (no labels at all)."""
    if neg == "old":
        return x[:, y == NEW], x[:, y != NEW]
    return x, hybrid_negatives(x, seed)


def ff_loss(model, x, y, neg, theta, seed):
    """Sum over layers of each layer's own loss; the layers share no gradient."""
    pos, negx = ff_split(x, y, neg, seed)
    k, th = CFG["cl"]["ff"]["kappa"], theta[:, None]
    zp = k * (ff_goodness(ff_counts(model, pos)) - th) / th
    zn = k * (ff_goodness(ff_counts(model, negx)) - th) / th
    return (F.softplus(-zp).mean(1) + F.softplus(zn).mean(1)).sum()


@torch.no_grad()
def ff_thresholds(model, neg):
    """Per-layer theta = the base network's mean goodness on positives and on negatives, averaged."""
    gp, gn = [], []
    for b, (wav, y) in enumerate(cl_batches(seed=4999)):
        if b >= NB_INIT:
            break
        pos, negx = ff_split(make_input(wav), y, neg, seed=b)
        gp.append(ff_goodness(ff_counts(model, pos)).mean(1))
        gn.append(ff_goodness(ff_counts(model, negx)).mean(1))
    return (0.5 * (torch.stack(gp).mean(0) + torch.stack(gn).mean(0))).clamp_min(1e-8)


def run_ff_arm(name):
    neg, kind = FF_ARMS[name]
    seed_all(CFG["seed"] + 79)
    model13 = expand_readout(MODEL, W_NEW0)
    frozen = frozen_set(kind)
    masks = plasticity_masks(model13, frozen, freeze_old_readout=True)
    hooks = attach_masks(model13, masks)
    ref = [lin.weight.detach().clone() for lin in model13.fc]
    theta = ff_thresholds(model13, neg)
    opt = torch.optim.Adam([lin.weight for lin in model13.fc[:-1]], lr=CFG["cl"]["lr"], weight_decay=0.0)
    W, lr_loc, norm_ref, curve = model13.fc[-1].weight, CFG["cl"]["local_lr"], None, []
    for ep in range(CFG["cl"]["epochs"]):
        model13.train()
        losses = []
        for b, (wav, y) in enumerate(cl_batches(seed=4000 + ep)):
            x = make_input(wav)
            loss = ff_loss(model13, x, y, neg, theta, seed=10007 * ep + b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            with torch.no_grad():                    # the new readout row: the same local rule as local_rule
                out = model13(x, record=True)
                traces = readout_trace(out["spikes"][-1], model13.beta)
                t_star = out["mem_out"][:, :, NEW].argmax(0)
                pre = traces[t_star, torch.arange(len(y), device=DEVICE)]
                if norm_ref is None:
                    norm_ref = float((pre ** 2).sum(1).mean().clamp_min(1e-6))
                W[NEW] -= (lr_loc / norm_ref) * (local_error(out["logits"], y)[:, None] * pre).mean(0)
        curve.append(epoch_row(model13, ep, losses, name))
    for h in hooks:
        h.remove()
    moved = max(float(((lin.weight.detach() - r).abs() * (1 - m)[:, None]).max()) for lin, r, m in zip(model13.fc, ref, masks))
    check(f"12 {name}: frozen weights did not move", moved == 0.0, f"largest change of a frozen weight {moved:.1e}")
    per_layer = [sum(model13.offsets[l] <= g < model13.offsets[l + 1] for g in frozen) for l in range(L_H)]
    return model13, curve, {"frozen_hidden": len(frozen), "frozen_per_layer": per_layer, "replay": False,
                            "negatives": neg, "theta": theta.tolist(), "lr": CFG["cl"]["lr"], "local_lr": lr_loc,
                            "old_readout_frozen": True}


def ff_locality_check(model13):
    """1) Layer 1's goodness sends no gradient into layer 0's weights (nothing crosses layers).
    2) Layer 0's gradient equals the closed form sum_t surrogate'(U[t] - thr) x[t] * 2 count / (N T^2 B),
    which uses only the current step (nothing flows through time)."""
    wav, _ = next(cl_batches(seed=33))
    x = make_input(wav)
    G = ff_goodness(ff_counts(model13, x)).mean(1)
    g_below = torch.autograd.grad(G[1], model13.fc[0].weight, retain_graph=True, allow_unused=True)[0]
    no_cross = g_below is None or float(g_below.abs().max()) == 0.0
    g_auto = torch.autograd.grad(G[0], model13.fc[0].weight)[0]
    with torch.no_grad():
        T, B, _ = x.shape
        N0 = model13.hidden[0]
        mem, cnt, sg = x.new_zeros(B, N0), x.new_zeros(B, N0), []
        for t in range(T):
            s, mem = lif_step(mem, model13.fc[0](x[t]), model13.beta, model13.thr)
            cnt += s
            sg.append(1.0 / (1.0 + (math.pi * (mem - model13.thr)) ** 2))
        wgt = 2.0 * cnt / (N0 * T_STEPS ** 2 * B)
        g_cf = sum((wgt * sg[t]).T @ x[t] for t in range(T))
    return no_cross, float(F.cosine_similarity(g_auto.flatten(), g_cf.flatten(), dim=0))
'''
set_src(C_ARMS, src(C_ARMS).rstrip("\n") + "\n" + FF_CODE)

# ---------------------------------------------------------------- run loop
replace_once(C_RUN, '''    ARM_LIST += ["slda_last", "slda_all"] if a == "slda" else [a]''',
             '''    ARM_LIST += ["slda_last", "slda_all"] if a == "slda" else list(COMBO_GROUPS) if a == "combo" else [a]''')
replace_once(C_RUN, '''    elif name in BPTT_ARMS:''', '''    elif name in FF_ARMS:
        model_arm, curve, info = run_ff_arm(name)
        no_cross, cos = ff_locality_check(model_arm)
        check(f"12 {name}: no gradient crosses layers or time steps", no_cross and cos > 0.999,
              f"layer 1 -> layer 0 gradient {'none' if no_cross else 'NONZERO'}; layer-0 gradient vs the one-step "
              f"closed form: cosine {cos:.6f}")
        info.update(locality_cosine=cos)
    elif name in BPTT_ARMS:''')

# ---------------------------------------------------------------- summary
replace_once(C_SUM, '''                 "sufficiency": ABL["sufficiency"], "union_keep_only": ABL["union_keep_only"]},
''', '''                 "sufficiency": ABL["sufficiency"], "union_keep_only": ABL["union_keep_only"]},
    "combo": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in COMBO.items()},
''')

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("patched", path, "cells:", len(cells))

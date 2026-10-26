"""Build snn_gscv2_tradeoff.ipynb from snn_gscv2_pipeline.ipynb: shared stages 0-11b copied, base training
switched to early stopping on validation loss, and new stages 12a-12e + 13 for the forgetting/plasticity study."""
import json, sys, uuid
from pathlib import Path

src_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
src = json.loads(src_path.read_text(encoding="utf-8"))
S = src["cells"]


def text(c):
    return "".join(c["source"])


def cell(kind, t):
    c = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": t.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def find(prefix):
    hits = [i for i, c in enumerate(S) if text(c).lstrip().startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


def patched(i, *pairs):
    t = text(S[i])
    for old, new in pairs:
        assert t.count(old) == 1, f"cell {i}: anchor found {t.count(old)} times: {old[:70]!r}"
        t = t.replace(old, new)
    return t


def copy(i):
    return cell(S[i]["cell_type"], text(S[i]))


I_CFG = find("# ---------------------------------------------------------------- what to run, and where")
I_HELP = find("# ---------------------------------------------------------------- small helpers used by every stage")
I_MD7 = find("## 7 · Base training")
I_TRAIN = find("def decide(logits):")
I_CHK7 = find("# ---- CHECK 7:")
I_MD12 = find("## 12 · Continual learning")
I_SETS = find('d12 = stage_dir("12_continual")')
I_CLH = find("NB_INIT, BS_CL = ")
I_ARMS = find("# ---- the arms")

cells = []

# ================================================================ intro
cells.append(cell("markdown", '''# Forgetting vs plasticity when a spiking keyword spotter learns a new word

This notebook continues `snn_gscv2_pipeline.ipynb`. It trains the same delta-encoded 3 × 512 spiking network on Google Speech Commands v2, finds each class's neurons (MI, Fisher, and blends of the two), and then studies **how much the old classes are forgotten versus how well "zero" is learned** under every knob that trades one for the other. Everything here is trained with BPTT.

| Stage | What it does | Saved in `RUN_DIR/` |
|---|---|---|
| 0–6 | Setup, data, front end, encoder, network (as in the pipeline notebook) | `config.json`, `log.txt`, `checks.json` |
| 7 | Base training with **early stopping**: stop once validation loss has not improved for 3 epochs in a row (cosine schedule over a 100-epoch cap); keep the lowest-validation-loss network | `07_train/` |
| 8–10 | Per-class Fisher, Omega, MI and neuron groups | `08_fisher/` … `10_groups/` |
| 11 / 11b | Ablation of the groups, and of blended groups for α = 0, 0.1, …, 1 in two rank modes | `11_ablation/`, `11b_combo/` |
| 12a | **How many epochs** to learn "zero": six approaches trained 20 epochs, scored every few epochs; E\\* picked on validation | `12a_epochs/` |
| 12b | Every regular approach and one masked approach per α, at E\\* | `12b_arms/` |
| 12c | **How many neurons to freeze**: 5–70% of the hidden neurons, chosen by MI, Fisher (global or within-layer), the best blend, or at random | `12c_budget/` |
| 12d | **Learning rate × replay fraction**, with no mask and with the best mask | `12d_tradeoff/` |
| 12e | The key comparison repeated over 3 seeds | `12e_seeds/` |
| 12f | **Stability first**: keep old-class accuracy within 2 points, then maximise "zero": output calibration, distillation, soft freezing (plain, Fisher- or blend-weighted), 80–95% freezing | `12f_stability/` |
| 12g | The best 12f settings (plus readout-only and 25% replay) repeated over seeds, as trained and calibrated | `12g_stability_seeds/` |
| 13 | Figures and `summary.json` | `summary.json`, `plots/` |

**Selection rule.** Every choice (E\\*, α\\*, the best freeze budget) is made on the **validation** split; every reported number is on the held-out **test** split.

**Trade-off measures.** *Forgetting*: drift (old-class accuracy drop with the new output ignored) and 13-class forgetting (including old clips the new output steals). *Plasticity*: "zero" accuracy and AUC. *Combined*: 13-class accuracy and **hmean** = harmonic mean of old-class accuracy and "zero" accuracy (13-class decisions), the criterion used for every selection.

**How to run.** Run once with `SMOKE_TEST = True` (synthetic data, minutes), then set it to `False`. `DATA_ROOT` points at the extracted GSC v2 folder; `CACHE_DIR` reuses the audio already decoded by the pipeline notebook. Each run of stage 12 is saved as it finishes, so an interrupted run resumes where it stopped. No internet access is needed.'''))
cells.append(copy(I_CFG - 1))       # "## 0 · Setup"
cells.append(copy(find("import os, sys, json, math")))

# ================================================================ config
CFG_CELL = '''# ---------------------------------------------------------------- what to run, and where
SMOKE_TEST = True     # True: tiny synthetic dataset, every stage in minutes. False: the real GSC v2.

DATA_ROOT = r"D:\\codes\\Anmol papers and code\\speech_commands_v2"   # folder that holds yes/, no/, ..., validation_list.txt
RUN_DIR   = r"./runs/gscv2_tradeoff"                                # every output of this notebook goes here
CACHE_DIR = r"./runs/gscv2_delta_3x512/cache"   # audio decoded by the pipeline notebook (reused); None -> RUN_DIR/cache

FORCE = set()     # stages to redo from scratch, e.g. {"07_train"}; every later stage is redone too

# ---------------------------------------------------------------- every hyperparameter
CFG = {
    "seed": 0,
    "data": {
        "keywords": ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"],
        "new_word": "zero",
        "new_word_in_base_unknown": True,
        "unknown_frac": 0.10,
        "silence_frac": 0.10,
        "sample_rate": 16000,
        "clip_len": 16000,
    },
    "augment": {"enabled": True, "shift_ms": 100.0, "noise_prob": 0.8, "noise_max_vol": 0.1},
    "frontend": {"n_fft": 400, "hop": 160, "n_mels": 64, "n_mfcc": 40, "f_min": 20.0, "f_max": 8000.0,
                 "mel_floor_min": 1e-3, "floor_batches": 10, "stats_batches": 40},
    "encoder": {"theta": 0.35},
    "net": {"hidden": [512, 512, 512], "beta": 0.9, "threshold": 1.0},
    "train": {"epochs": 100,                  # a cap: early stopping normally ends training first
              "early_stop_patience": 3,       # stop after this many epochs in a row without a lower validation loss
              "batch_size": 128, "lr": None,  # None -> 5e-4 * 256 / widest layer; cosine decay over the cap
              "val_every": 1, "max_train_batches": None, "eval_batch_size": 256},
    "analysis": {"topk": 40, "ridge": 1e-2, "mi_bins": 8, "mi_shuffles": 20, "random_reps": 5,
                 "ksweep": [20, 40, 80, 160, 320], "save_full_fisher": True},
    "combo": {"alphas": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],   # score = alpha * MI rank + (1 - alpha) * Fisher rank
              "rank_modes": ["global", "within_layer"]},
    "cl": {"batch_size": 128, "lr": None,     # None -> the base learning rate
           "local_lr": 0.1, "slda_shrinkage": 1e-2, "k_new_neurons": 64, "init_batches": 20,
           "epochs": 5, "ff": {"kappa": 4.0, "hybrid_block": 10}},    # (epochs / ff: used only by helper code kept from the pipeline)
    "tradeoff": {
        "epoch_sweep": {"max_epochs": 20, "eval_epochs": [1, 2, 3, 5, 8, 12, 16, 20],
                        "arms": ["no_mask", "mi_mask", "fisher_mask", "combo_layer_a050", "no_mask_replay", "readout_only"]},
        "budgets": [0.05, 0.10, 0.20, 0.30, 0.50, 0.70],        # share of all hidden neurons frozen
        "budget_methods": ["mi", "fisher_global", "fisher_layer", "blend_best", "random"],
        "lr_mults": [0.25, 1.0, 4.0],                            # x the continual-learning learning rate
        "replay_fracs": [0.0, 0.01, 0.05, 0.25, 0.5],            # share of each batch that is LABELLED old clips (0.5 = all old clips)
        "seeds": [0, 1, 2],
    },
    "stability": {                                       # stage 12f: stability first
        "max_old_drop_pts": 2.0,                         # old-class accuracy may fall at most this much (validation)
        "lwf_lambdas": [0.3, 1.0, 3.0, 10.0, 30.0],      # distillation strength
        "lwf_T": 2.0,                                    # distillation temperature
        "soft_importance": ["uniform", "fisher", "blend"],
        "soft_lambdas": [10.0, 100.0, 1000.0, 10000.0, 100000.0],   # soft-freezing strength (on relative weight change)
        "extreme_methods": ["blend_best", "random"],
        "extreme_budgets": [0.8, 0.9, 0.95],
    },
    "stability_seeds": {                                 # stage 12g: seeds for the stability-first winners
        "top_k": 4,                                      # best 12f settings within the limit (validation)
        "seeds": [1, 2],                                 # seed 0 is the 12f run itself
        "references": ["ref_readout_only", "ref_replay_25"],
    },
}

if SMOKE_TEST:
    RUN_DIR = "./runs/smoke_tradeoff"
    DATA_ROOT = str(Path(RUN_DIR) / "synthetic_gsc")
    CACHE_DIR = None
    CFG["net"]["hidden"] = [64, 64, 64]
    CFG["train"].update(epochs=30, batch_size=32, eval_batch_size=64)
    CFG["frontend"].update(floor_batches=3, stats_batches=4)
    CFG["analysis"].update(topk=8, mi_shuffles=5, random_reps=2, ksweep=[4, 8, 16, 32])
    CFG["cl"].update(batch_size=32, k_new_neurons=8, init_batches=3)
    CFG["tradeoff"]["epoch_sweep"].update(max_epochs=6, eval_epochs=[1, 2, 3, 4, 6])
    CFG["tradeoff"].update(seeds=[0, 1], budgets=[0.1, 0.5], lr_mults=[1.0, 4.0], replay_fracs=[0.0, 0.5])
    CFG["combo"]["alphas"] = [0.0, 0.5, 1.0]
    CFG["stability"].update(lwf_lambdas=[1.0, 10.0], soft_lambdas=[10.0, 1000.0], extreme_budgets=[0.9])
    CFG["stability_seeds"]["seeds"] = [1]

if CFG["train"]["lr"] is None:
    CFG["train"]["lr"] = 5e-4 * 256 / max(CFG["net"]["hidden"])
if CFG["cl"]["lr"] is None:
    CFG["cl"]["lr"] = CFG["train"]["lr"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RUN = Path(RUN_DIR)
RUN.mkdir(parents=True, exist_ok=True)
CACHE = Path(CACHE_DIR) if CACHE_DIR else RUN / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
print("device:", DEVICE, "| run dir:", RUN.resolve(), "| cache:", CACHE.resolve(), "| lr:", CFG["train"]["lr"])
if DEVICE == "cpu" and not SMOKE_TEST:
    print("warning: PyTorch sees no CUDA GPU, so the real run will train on the CPU and take many hours.")
'''
cells.append(cell("code", CFG_CELL))

# ================================================================ helpers (new stage list)
cells.append(cell("code", patched(I_HELP,
    ('''          "08_fisher", "09_mi", "10_groups", "11_ablation", "11b_combo", "12_continual"]''',
     '''          "08_fisher", "09_mi", "10_groups", "11_ablation", "11b_combo",
          "12a_epochs", "12b_arms", "12c_budget", "12d_tradeoff", "12e_seeds", "12f_stability",
          "12g_stability_seeds"]'''),
    ('''"08_fisher": ["analysis"], "11b_combo": ["combo"],
             "12_continual": ["cl"]}''',
     '''"08_fisher": ["analysis"], "11b_combo": ["combo"],
             "12a_epochs": ["cl", "tradeoff"], "12f_stability": ["stability"],
             "12g_stability_seeds": ["stability_seeds"]}'''))))

# ================================================================ stages 0b-6 unchanged
for i in range(I_HELP + 1, I_MD7):
    cells.append(copy(i))

# ================================================================ stage 7: early stopping on validation loss
cells.append(cell("markdown", '''## 7 · Base training (BPTT) with early stopping

Adam at lr = 5e-4 · 256 / 512 = 2.5e-4, cosine decay over a **100-epoch cap**, batch 128, augmentation on. After every epoch the network is scored on **validation**, now including the validation **loss** (cross-entropy of the max-membrane logits). Training **stops once the validation loss has not improved for 3 epochs in a row**, and the network with the lowest validation loss is kept as `best.pt`. `last.pt` is written every epoch with the optimiser state and the early-stopping counter, so an interrupted run resumes where it stopped.

**Check:** as in the pipeline notebook, plus whether training ended by early stopping or by the cap.'''))

NEW_TRAIN_BASE = '''def train_base():
    """Adam with a cosine schedule over the epoch cap. Stops once the validation LOSS has not improved for
    `early_stop_patience` epochs in a row; the lowest-validation-loss network is kept as best.pt."""
    cfgt = CFG["train"]
    patience = cfgt["early_stop_patience"]
    seed_all(CFG["seed"])
    model = new_model()
    opt = torch.optim.Adam(model.parameters(), lr=cfgt["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfgt["epochs"])
    start, history, best_loss, bad = 0, [], float("inf"), 0
    basis = stage_basis("07_train")
    if (d7 / "last.pt").exists() and "07_train" not in FORCE:
        ck = load_pt(d7 / "last.pt")
        if ck.get("basis") == basis:
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start, history, best_loss, bad = ck["epoch"] + 1, ck["history"], ck["best_loss"], ck["bad"]
            log(f"07_train: resuming after epoch {start} ({bad} epoch(s) without improvement so far)")
        else:
            log("07_train: last.pt was trained with other settings or inputs; starting over")
    for epoch in range(start, cfgt["epochs"]):
        if bad >= patience:
            break
        row = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], **train_one_epoch(model, opt, epoch)}
        ev = evaluate(model, VAL12)
        row.update(val_acc=ev["acc"], val_loss=ev["loss"])
        if ev["loss"] < best_loss:
            best_loss, bad, row["best"] = ev["loss"], 0, True
            save_pt({"model": model.state_dict(), "epoch": epoch + 1, "val_acc": ev["acc"], "val_loss": ev["loss"],
                     "cfg": CFG}, d7 / "best.pt")
        else:
            bad += 1
        sched.step()
        history.append(row)
        save_json(history, d7 / "history.json")
        save_pt({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
                 "history": history, "best_loss": best_loss, "bad": bad, "basis": basis}, d7 / "last.pt")
        log(f"  ep {epoch + 1:3d}/{cfgt['epochs']} | loss {row['loss']:.3f} | train {row['train_acc']:.3f} | val loss "
            f"{ev['loss']:.4f} acc {ev['acc']:.3f} (best loss {best_loss:.4f}; {bad} without improvement) | "
            f"rates {row['rates']} | {row['sec']:.0f} s, {row['clips_per_s']:.0f} clips/s")
        if len(history) >= 2 and all(abs(r["loss"] - math.log(12)) < 0.01 for r in history[-2:]):
            check("07 training did not stall at ln(12)", False, "loss stuck at 2.485: every logit is equal. Lower the learning rate.")
            raise RuntimeError("training stalled at ln(12): every logit is equal. Lower CFG['train']['lr'].")
        if min(row["rates"]) < 1e-5:
            log("  warning: a hidden layer is (almost) silent; no gradient flows through a silent layer")
        if bad >= patience:
            log(f"  early stop after epoch {epoch + 1}: validation loss has not improved for {patience} epochs "
                f"(best {best_loss:.4f})")
    check("07 training did not stall at ln(12)", True, f"{len(history)} epochs, final loss {history[-1]['loss']:.3f}")
    return history


'''
t = text(S[I_TRAIN])
a, b = t.index("def train_base():"), t.index('if is_done("07_train"):')
t = t[:a] + NEW_TRAIN_BASE + t[b:]
for old, new in [
    ('''    hit_argmax = 0
    for wav, y in data.batches(batch_size or CFG["train"]["eval_batch_size"]):
        logits = model(make_input(wav), masks=masks)["logits"]
''', '''    hit_argmax, loss_sum, n_loss = 0, 0.0, 0
    for wav, y in data.batches(batch_size or CFG["train"]["eval_batch_size"]):
        logits = model(make_input(wav), masks=masks)["logits"]
        yd = y.to(logits.device)
        known = yd < logits.shape[1]                      # a 12-output net cannot be scored on label 12
        if bool(known.any()):
            loss_sum += float(F.cross_entropy(logits[known].float(), yd[known], reduction="sum"))
            n_loss += int(known.sum())
'''),
    ('''"acc_argmax": hit_argmax / n}''', '''"acc_argmax": hit_argmax / n,
            "loss": loss_sum / max(n_loss, 1)}'''),
    ('''    """Accuracy, per-class accuracy and confusion matrix [true, predicted] on a ClipSet, in a fixed order.''',
     '''    """Accuracy, per-class accuracy, confusion matrix [true, predicted] and mean cross-entropy on a ClipSet.'''),
]:
    assert t.count(old) == 1, old[:60]
    t = t.replace(old, new)
cells.append(cell("code", t))
cells.append(cell("code", text(S[I_CHK7]) + '''
stopped_early = len(H) < CFG["train"]["epochs"]
best_ep = min(H, key=lambda r: r.get("val_loss", float("inf")))["epoch"]
print(f"\\ntraining ran {len(H)} epochs ({'early stop' if stopped_early else 'reached the cap'}); lowest validation loss "
      f"{min(r['val_loss'] for r in H):.4f} at epoch {best_ep}")
check("07 training ended by early stopping, not the epoch cap (a finding, not a bug, if it fails)", stopped_early,
      f"{len(H)} of {CFG['train']['epochs']} epochs; best validation loss at epoch {best_ep}")
'''))

# ================================================================ stages 8-11b unchanged
for i in range(I_CHK7 + 1, I_MD12):
    cells.append(copy(i))

# ================================================================ stage 12: shared setup + helpers from the pipeline
cells.append(cell("markdown", '''## 12 · Forgetting vs plasticity: learning "zero"

Every run here starts from the same base network, adds a 13th output for "zero" and trains it with BPTT. Batches are half "zero" clips and half old clips. The 12 old output rows stay frozen in every run, so runs differ only in **which hidden neurons may change** (their incoming weights frozen or not), **how long** they train, the **learning rate**, and how many old clips carry their **label** (replay):

- **replay fraction 0:** old clips are unlabelled; they only keep the new output below the best old output (softplus term), as in the pipeline notebook's no-replay arms.
- **replay fraction f > 0:** a share f of the whole batch is labelled old clips and gets the 13-way cross-entropy. f = 0.5 labels every old clip, which is the pipeline's full replay.

The helpers below (data sets, the new-row initialisation, masks, scoring) are the pipeline notebook's stage-12 code.'''))
cells.append(cell("code", patched(I_SETS, ('d12 = stage_dir("12_continual")', 'd12 = stage_dir("12_setup")'))))
cells.append(copy(I_CLH))
cells.append(copy(I_ARMS))

RUNNER = r'''# ---- the shared runner: one continual-learning run, scored on validation and test
TR = CFG["tradeoff"]
f_init = d12 / "new_row_init.pt"
BASIS_INIT = stage_basis("12a_epochs")
saved_init = load_pt(f_init) if f_init.exists() and "12a_epochs" not in FORCE else {}
if saved_init.get("basis") == BASIS_INIT:
    W_NEW0 = saved_init["w"].to(DEVICE)
else:
    W_NEW0, init_info = init_new_row(MODEL)
    save_pt({"w": W_NEW0.cpu(), "basis": BASIS_INIT, **init_info}, f_init)
    log(f"  new readout row initialised: {init_info}")

KEEP = ["acc_13", "old_mean_acc", "hmean", "forgetting_mean_pts", "forgetting_pts", "drift_forgetting_mean_pts",
        "drift_forgetting_pts", "new_word_acc", "new_word_auc", "new_word_auc_vs_class", "old_clips_called_new",
        "new_word_to_unknown", "per_class", "no_decision"]


def hmean(old, new):
    """Harmonic mean of old-class accuracy and new-word accuracy: high only when both are high."""
    return 2.0 * old * new / max(old + new, 1e-12)


def scores(model13, data, base_old):
    s = summarize(*logits_of(model13, data), base_old)
    s["hmean"] = hmean(s["old_mean_acc"], s["new_word_acc"])
    return {k: s[k] for k in KEEP if k in s}


def cl_batches_mix(seed, replay_frac):
    """The pipeline's cl_batches, plus which old clips keep their label. A share replay_frac of the whole batch is
    labelled old clips (each old clip is labelled with probability 2 * replay_frac)."""
    gen = torch.Generator().manual_seed(seed + 7919)
    for wav, y in cl_batches(seed):
        lab = (y != NEW) & (torch.rand(len(y), generator=gen) < 2 * replay_frac).to(y.device)
        yield wav, y, lab


def cl_loss_mix(logits, y, lab):
    """13-way cross-entropy on new-word clips and labelled old clips; unlabelled old clips only keep the new logit
    below the best old logit. replay_frac 0 and 0.5 reproduce the pipeline's no-replay and replay losses."""
    sup = (y == NEW) | lab
    loss = logits.new_zeros(())
    if bool(sup.any()):
        loss = loss + F.cross_entropy(logits[sup], y[sup])
    if bool((~sup).any()):
        z = logits[~sup]
        loss = loss + F.softplus(z[:, NEW] - z[:, :NEW].max(-1).values).mean()
    return loss


def train_cl(frozen, replay_frac=0.0, lr_mult=1.0, epochs=None, seed=0, eval_epochs=(), tag="",
             lwf=0.0, soft=None, keep_logits=False):
    """One run. Returns validation and test scores at the end (and at eval_epochs), the validation curve,
    how far each layer's weights moved, and the largest change of any frozen weight (must be 0).
    lwf > 0 adds distillation: the 12 old logits are kept close to the base network's on every clip of the batch
    (no labels needed). soft = {"weights": per-layer importance vectors, "lam": strength} adds a penalty on how far
    each neuron's incoming weights move, weighted by that neuron's importance (soft freezing).
    keep_logits=True also returns the final validation and test logits (for calibration)."""
    epochs = epochs or E_STAR
    seed_all(CFG["seed"] + 77 + 1000 * seed)
    model13 = expand_readout(MODEL, W_NEW0)
    masks = plasticity_masks(model13, frozen, freeze_old_readout=True)
    hooks = attach_masks(model13, masks)
    ref = [lin.weight.detach().clone() for lin in model13.fc]
    ref_sq = sum(float((r ** 2).sum()) for r in ref[:L_H])
    opt = torch.optim.Adam(model13.parameters(), lr=CFG["cl"]["lr"] * lr_mult, weight_decay=0.0)
    T_lwf = CFG.get("stability", {}).get("lwf_T", 2.0)
    curve, snaps = [], {}
    for ep in range(epochs):
        model13.train()
        losses = []
        for wav, y, lab in cl_batches_mix(1000 + ep + 100003 * seed, replay_frac):
            x = make_input(wav)
            z = model13(x)["logits"]
            loss = cl_loss_mix(z, y, lab)
            if lwf > 0:
                with torch.no_grad():
                    zb = MODEL(x)["logits"]
                loss = loss + lwf * T_lwf ** 2 * F.kl_div(F.log_softmax(z[:, :NEW] / T_lwf, -1), F.softmax(zb / T_lwf, -1),
                                                         reduction="batchmean")
            if soft is not None:
                pen = sum((w[:, None] * (model13.fc[l].weight - ref[l]) ** 2).sum() for l, w in enumerate(soft["weights"]))
                loss = loss + soft["lam"] * pen / ref_sq
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        v = scores(model13, VAL13, BASE_OLD_VAL)
        curve.append({"epoch": ep + 1, "loss": float(np.mean(losses)), **{k: v[k] for k in (
            "drift_forgetting_mean_pts", "forgetting_mean_pts", "old_mean_acc", "new_word_acc", "new_word_auc", "hmean", "acc_13")}})
        if ep + 1 in eval_epochs:
            snaps[ep + 1] = {"val": v, "test": scores(model13, TEST13, BASE_OLD)}
            log(f"    {tag} ep {ep + 1}: val hmean {v['hmean']:.3f}, drift {v['drift_forgetting_mean_pts']:+.1f} | "
                f"test hmean {snaps[ep + 1]['test']['hmean']:.3f}")
    for h in hooks:
        h.remove()
    moved = max(float(((lin.weight.detach() - r).abs() * (1 - m)[:, None]).max()) for lin, r, m in zip(model13.fc, ref, masks))
    rel = [float((model13.fc[l].weight.detach() - ref[l]).norm() / ref[l].norm()) for l in range(L_H)]
    out = {"val": v, "test": snaps[epochs]["test"] if epochs in snaps else scores(model13, TEST13, BASE_OLD),
           "curve": curve, "snaps": {str(k): s for k, s in snaps.items()}, "frozen_moved": moved,
           "weight_change": rel, "frozen_per_layer": [sum(MODEL.offsets[l] <= g < MODEL.offsets[l + 1] for g in frozen)
                                                      for l in range(L_H)],
           "settings": {"frozen": len(frozen), "replay_frac": replay_frac, "lr_mult": lr_mult, "epochs": epochs, "seed": seed,
                        "lwf": lwf, "soft_lam": None if soft is None else soft["lam"]}}
    if keep_logits:
        Zv, Yv = logits_of(model13, VAL13)
        Zt, Yt = logits_of(model13, TEST13)
        out["logits"] = {"val_Z": Zv, "val_Y": Yv, "test_Z": Zt, "test_Y": Yt}
    del model13, opt
    return out


def run_cached(stage, name, fn):
    """Run fn() once and save its result; a later run with the same settings and inputs loads it instead."""
    d = stage_dir(stage) / "runs"
    d.mkdir(exist_ok=True)
    f, basis = d / f"{name}.json", stage_basis(stage)
    if f.exists() and stage not in FORCE:
        r = load_json(f)
        if r.get("basis") == basis:
            return r
    t0 = time.time()
    r = fn()
    r.update(name=name, basis=basis, seconds=round(time.time() - t0))
    save_json(r, f)
    te = r["test"]
    log(f"{stage} {name}: test drift {te['drift_forgetting_mean_pts']:+.2f} | 13-way forgetting {te['forgetting_mean_pts']:+.2f} | "
        f"'{NEW_WORD}' acc {te['new_word_acc']:.3f} AUC {te['new_word_auc']:.3f} | hmean {te['hmean']:.3f} "
        f"(val {r['val']['hmean']:.3f}) | {r['seconds']} s")
    return r


def frozen_ok(stage, runs):
    worst = max((r.get("frozen_moved", 0.0) for r in runs.values()), default=0.0)
    check(f"{stage}: frozen weights never moved, in all {len(runs)} runs", worst == 0.0, f"largest change {worst:.1e}")


def table(runs, keys=("drift_forgetting_mean_pts", "forgetting_mean_pts", "new_word_acc", "new_word_auc", "hmean", "acc_13")):
    print(f"{'run':<34}{'val hmean':>10}" + "".join(f"{k.replace('_mean_pts', '').replace('_', ' '):>18}" for k in keys))
    for n, r in runs.items():
        print(f"{n:<34}{r['val']['hmean']:>10.3f}" + "".join(f"{r['test'][k]:>18.3f}" for k in keys))


ARMS_REG = {                        # name: (frozen-set kind, replay fraction)
    "no_mask": ("none", 0.0), "mi_mask": ("mi", 0.0), "random_mask": ("random", 0.0),
    "random_layer_matched": ("random_layer_matched", 0.0), "fisher_mask": ("fisher", 0.0),
    "layer0_only": ("layer0", 0.0), "mi_plus_layer0": ("mi+layer0", 0.0), "mi_plus_fisher": ("mi+fisher", 0.0),
    "readout_only": ("all", 0.0), "no_mask_replay": ("none", 0.5), "mi_mask_replay": ("mi", 0.5)}
for _name in COMBO_GROUPS:
    ARMS_REG[_name] = ("combo:" + _name, 0.0)
'''
cells.append(cell("code", RUNNER))

# ---------------------------------------------------------------- 12a
cells.append(cell("markdown", '''### 12a · How many epochs to learn "zero"

Six approaches (no mask, MI mask, Fisher mask, the within-layer α = 0.5 blend, no mask + full replay, readout only) are trained for 20 epochs and scored on validation and test at epochs 1, 2, 3, 5, 8, 12, 16 and 20. **E\\*** is the epoch with the highest mean validation hmean over the six; every later stage trains for E\\* epochs.

**Check:** frozen weights never moved; E\\* is one of the scored epochs.'''))
cells.append(cell("code", r'''EP = TR["epoch_sweep"]
EPOCH_RUNS = {}
for a in EP["arms"]:
    kind, rf = ARMS_REG[a]
    EPOCH_RUNS[a] = run_cached("12a_epochs", a, lambda kind=kind, rf=rf, a=a: train_cl(
        frozen_set(kind), replay_frac=rf, epochs=EP["max_epochs"], eval_epochs=EP["eval_epochs"], tag=a))
EPOCH_TABLE = {e: {a: r["snaps"][str(e)] for a, r in EPOCH_RUNS.items()} for e in EP["eval_epochs"]}
mean_val = {e: float(np.mean([s["val"]["hmean"] for s in EPOCH_TABLE[e].values()])) for e in EP["eval_epochs"]}
E_STAR = max(EP["eval_epochs"], key=lambda e: (mean_val[e], -e))
if not is_done("12a_epochs"):
    mark_done("12a_epochs", e_star=E_STAR)
log(f"12a_epochs: E* = {E_STAR} (mean validation hmean {mean_val[E_STAR]:.3f})")
'''))
cells.append(cell("code", r'''# ---- CHECK 12a: the epoch table
frozen_ok("12a", EPOCH_RUNS)
check("12a E* is one of the scored epochs, chosen on validation", E_STAR in EP["eval_epochs"], f"E* = {E_STAR}")
print(f"{'epoch':>6}{'mean val hmean':>16}   test hmean / test drift per approach")
for e in EP["eval_epochs"]:
    print(f"{e:>6}{mean_val[e]:>16.3f}   " + " | ".join(f"{a} {s['test']['hmean']:.3f}/{s['test']['drift_forgetting_mean_pts']:+.1f}"
                                                  for a, s in EPOCH_TABLE[e].items()))
print("best validation epoch per approach:",
      {a: max(EP["eval_epochs"], key=lambda e: EPOCH_TABLE[e][a]["val"]["hmean"]) for a in EP["arms"]})
if HAVE_PLT:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
    for a in EP["arms"]:
        es = EP["eval_epochs"]
        axes[0].plot(es, [EPOCH_TABLE[e][a]["val"]["hmean"] for e in es], marker="o", label=a)
        axes[1].plot(es, [EPOCH_TABLE[e][a]["val"]["drift_forgetting_mean_pts"] for e in es], marker="o")
        axes[2].plot(es, [EPOCH_TABLE[e][a]["val"]["new_word_acc"] for e in es], marker="o")
    for ax, t in zip(axes, ["validation hmean", "validation drift (pts)", f"validation '{NEW_WORD}' accuracy"]):
        ax.axvline(E_STAR, ls="--", c="grey", lw=0.8)
        ax.set_title(t)
        ax.set_xlabel("new-word epochs")
    axes[0].legend(fontsize=7)
    savefig("12a_epochs")
'''))

# ---------------------------------------------------------------- 12b
cells.append(cell("markdown", '''### 12b · Regular approaches and the α sweep, at E\\* epochs

Every approach from the pipeline notebook (no mask, MI, random, layer-matched random, Fisher, layer 0 only, MI + layer 0, MI + Fisher, readout only, replay with and without the MI mask) plus one masked approach per stage-11b group set (11 α values × 2 rank modes), and the SLDA head as a backprop-free reference. **α\\*** is the blend with the highest validation hmean; stage 12c uses it.'''))
cells.append(cell("code", r'''ARM_RUNS = {}
for a, (kind, rf) in ARMS_REG.items():
    ARM_RUNS[a] = run_cached("12b_arms", a, lambda kind=kind, rf=rf, a=a: train_cl(frozen_set(kind), replay_frac=rf, tag=a))


def slda_run():
    m = run_slda_arms()["slda_all"][0]
    return {"val": scores(m, VAL13, BASE_OLD_VAL), "test": scores(m, TEST13, BASE_OLD), "frozen_moved": 0.0}


SLDA_RUN = run_cached("12b_arms", "slda_all", slda_run)
BLENDS = {a: r for a, r in ARM_RUNS.items() if a.startswith("combo_")}
BEST_BLEND = max(BLENDS, key=lambda a: BLENDS[a]["val"]["hmean"])
BEST_BLEND_MODE, BEST_ALPHA = COMBO_META[BEST_BLEND]["mode"], COMBO_META[BEST_BLEND]["alpha"]
if not is_done("12b_arms"):
    mark_done("12b_arms", best_blend=BEST_BLEND)
log(f"12b_arms: best blend on validation = {BEST_BLEND} (mode {BEST_BLEND_MODE}, alpha {BEST_ALPHA})")
'''))
cells.append(cell("code", r'''# ---- CHECK 12b: results tables
frozen_ok("12b", ARM_RUNS)
check("12b alpha* chosen on validation among the blended masks", BEST_BLEND in BLENDS, BEST_BLEND)
table({**{a: r for a, r in ARM_RUNS.items() if not a.startswith("combo_")}, "slda_all": SLDA_RUN})
print()
table(BLENDS)
if HAVE_PLT:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for mode in CFG["combo"]["rank_modes"]:
        rs = sorted((a for a in BLENDS if COMBO_META[a]["mode"] == mode), key=lambda a: COMBO_META[a]["alpha"])
        xs = [COMBO_META[a]["alpha"] for a in rs]
        axes[0].plot(xs, [BLENDS[a]["val"]["hmean"] for a in rs], marker="o", label=mode)
        axes[1].plot(xs, [BLENDS[a]["test"]["drift_forgetting_mean_pts"] for a in rs], marker="o", label=mode)
    axes[0].set_title("validation hmean by alpha")
    axes[1].set_title("test drift by alpha (pts)")
    for ax in axes:
        ax.set_xlabel("alpha (0 = Fisher, 1 = MI)")
        ax.legend()
    savefig("12b_alpha")
'''))

# ---------------------------------------------------------------- 12c
cells.append(cell("markdown", '''### 12c · How many neurons to freeze

For each selection method, a **budget** of 5, 10, 20, 30, 50 or 70% of all hidden neurons is frozen. Neurons are taken **round-robin over the 12 classes**, each class contributing its next most important neuron in turn, so every class gets a fair share and a larger budget always contains the smaller ones. Methods: per-class **MI**; per-class **Fisher** ranked over the whole network (`fisher_global`, depth-biased) or within each layer (`fisher_layer`); the **best blend** from 12b; and a size-matched **random** set. The best (method, budget) on validation hmean is used in 12d and 12e.'''))
cells.append(cell("code", r'''def class_rankings(method):
    """Per class, every hidden neuron from most to least important."""
    wl = BEST_BLEND_MODE == "within_layer"
    if method == "mi":
        S = MI["per_class"].double()
    elif method == "fisher_global":
        S = FISH["naive"].double()
    elif method == "fisher_layer":
        S = rank_scores(FISH["naive"], True)
    elif method == "blend_best":
        S = BEST_ALPHA * rank_scores(MI["per_class"], wl) + (1 - BEST_ALPHA) * rank_scores(FISH["naive"], wl)
    else:
        raise ValueError(method)
    S = S + TIEBREAK
    return [torch.argsort(S[c], descending=True).tolist() for c in range(S.shape[0])]


def budget_set(method, B):
    """B neurons, picked round-robin over classes (nested: a larger budget contains every smaller one)."""
    if method == "random":
        return set(np.random.default_rng(CFG["seed"] + 555).permutation(N_H)[:B].tolist())
    lists, out, r = class_rankings(method), [], 0
    seen = set()
    while len(out) < B:
        for lst in lists:
            if len(out) >= B:
                break
            if lst[r] not in seen:
                seen.add(lst[r])
                out.append(lst[r])
        r += 1
    return set(out)


BUDGETS = {frac: max(1, int(round(frac * N_H))) for frac in TR["budgets"]}
BUDGET_RUNS = {}
for m in TR["budget_methods"]:
    for frac, B in BUDGETS.items():
        name = f"{m}_b{int(round(100 * frac)):02d}"
        BUDGET_RUNS[name] = run_cached("12c_budget", name, lambda m=m, B=B, name=name: {
            **train_cl(budget_set(m, B), tag=name), "method": m, "budget": B, "budget_frac": frac})
selectable = {n: r for n, r in BUDGET_RUNS.items() if r["method"] != "random"}
BEST_MASK = max(selectable, key=lambda n: selectable[n]["val"]["hmean"])
BEST_METHOD, BEST_B = BUDGET_RUNS[BEST_MASK]["method"], BUDGET_RUNS[BEST_MASK]["budget"]
if not is_done("12c_budget"):
    mark_done("12c_budget", best_mask=BEST_MASK)
log(f"12c_budget: best mask on validation = {BEST_MASK} ({BEST_B} neurons)")
'''))
cells.append(cell("code", r'''# ---- CHECK 12c: budget curves
frozen_ok("12c", BUDGET_RUNS)
small, big = min(BUDGETS.values()), max(BUDGETS.values())
check("12c budgets are nested (the smallest set sits inside the largest) for every method",
      all(budget_set(m, small) <= budget_set(m, big) for m in TR["budget_methods"]))
check("12c each budget freezes exactly B neurons", all(r["settings"]["frozen"] == r["budget"] for r in BUDGET_RUNS.values()))
table(BUDGET_RUNS)
if HAVE_PLT:
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for m in TR["budget_methods"]:
        rs = sorted((r for r in BUDGET_RUNS.values() if r["method"] == m), key=lambda r: r["budget"])
        xs = [100 * r["budget_frac"] for r in rs]
        axes[0].plot(xs, [r["test"]["drift_forgetting_mean_pts"] for r in rs], marker="o", label=m)
        axes[1].plot(xs, [r["test"]["new_word_acc"] for r in rs], marker="o", label=m)
        axes[2].plot(xs, [r["val"]["hmean"] for r in rs], marker="o", label=m)
    for ax, t in zip(axes, ["test drift (pts)", f"test '{NEW_WORD}' accuracy", "validation hmean"]):
        ax.set_title(t)
        ax.set_xlabel("% of hidden neurons frozen")
    axes[0].legend(fontsize=8)
    savefig("12c_budget")
'''))

# ---------------------------------------------------------------- 12d
cells.append(cell("markdown", '''### 12d · Learning rate × replay fraction

The two remaining dials, crossed: the continual-learning **learning rate** (0.25×, 1×, 4×) and the **replay fraction** (0, 1, 5, 25, 50% of each batch as labelled old clips), each with **no mask** and with the **best mask** from 12c. This maps the forgetting–plasticity frontier: every run is a point (old-class accuracy, "zero" accuracy).'''))
cells.append(cell("code", r'''MASKS_12D = {"none": set(), "best": budget_set(BEST_METHOD, BEST_B)}
TRADE_RUNS = {}
for mk, fz in MASKS_12D.items():
    for lm in TR["lr_mults"]:
        for rf in TR["replay_fracs"]:
            name = f"{mk}_lr{lm:g}_rp{rf:g}"
            TRADE_RUNS[name] = run_cached("12d_tradeoff", name, lambda fz=fz, lm=lm, rf=rf, name=name, mk=mk: {
                **train_cl(fz, replay_frac=rf, lr_mult=lm, tag=name), "mask": mk, "lr_mult": lm, "replay_frac": rf})
if not is_done("12d_tradeoff"):
    mark_done("12d_tradeoff")
'''))
cells.append(cell("code", r'''# ---- CHECK 12d: the frontier
frozen_ok("12d", TRADE_RUNS)
table(TRADE_RUNS)


def pareto(points):
    """Indices of points not beaten on both coordinates (higher is better on both)."""
    return [i for i, (x, y) in enumerate(points) if not any(x2 >= x and y2 >= y and (x2, y2) != (x, y) for x2, y2 in points)]


ALL_RUNS = {**{f"12b/{k}": v for k, v in ARM_RUNS.items()}, **{f"12c/{k}": v for k, v in BUDGET_RUNS.items()},
            **{f"12d/{k}": v for k, v in TRADE_RUNS.items()}}
names = list(ALL_RUNS)
pts = [(ALL_RUNS[n]["test"]["old_mean_acc"], ALL_RUNS[n]["test"]["new_word_acc"]) for n in names]
FRONT = [names[i] for i in pareto(pts)]
print("\nPareto front (test old-class accuracy vs test new-word accuracy):")
for n in sorted(FRONT, key=lambda n: -ALL_RUNS[n]["test"]["old_mean_acc"]):
    t = ALL_RUNS[n]["test"]
    print(f"  {n:<40} old {t['old_mean_acc']:.3f}  new {t['new_word_acc']:.3f}  hmean {t['hmean']:.3f}")
if HAVE_PLT:
    plt.figure(figsize=(7, 5))
    for grp, mk in [("12b", "o"), ("12c", "s"), ("12d", "^")]:
        sel = [n for n in names if n.startswith(grp)]
        plt.scatter([ALL_RUNS[n]["test"]["old_mean_acc"] for n in sel], [ALL_RUNS[n]["test"]["new_word_acc"] for n in sel],
                    marker=mk, s=18, alpha=0.7, label=grp)
    fr = sorted(FRONT, key=lambda n: ALL_RUNS[n]["test"]["old_mean_acc"])
    plt.plot([ALL_RUNS[n]["test"]["old_mean_acc"] for n in fr], [ALL_RUNS[n]["test"]["new_word_acc"] for n in fr], "k--", lw=0.8)
    plt.axvline(float(BASE_OLD.mean()), c="grey", lw=0.8, ls=":")
    plt.xlabel("old-class accuracy, 13-way (test)")
    plt.ylabel(f"'{NEW_WORD}' accuracy (test)")
    plt.legend()
    plt.title("forgetting vs plasticity (dotted: base network)")
    savefig("12d_frontier")
'''))

# ---------------------------------------------------------------- 12e
cells.append(cell("markdown", '''### 12e · Seeds

The key comparison is repeated with every seed in `CFG["tradeoff"]["seeds"]`. The base network is the same; the seed changes the order of the continual-learning batches and the augmentation (each mask stays the same set of neurons). Reported as mean ± SD.'''))
cells.append(cell("code", r'''KEY = {"no_mask": set(), "mi_mask": frozen_set("mi"), "fisher_mask": frozen_set("fisher"),
       f"best ({BEST_MASK})": budget_set(BEST_METHOD, BEST_B), f"random ({BEST_B})": budget_set("random", BEST_B),
       "no_mask_replay": "replay", "readout_only": frozen_set("all")}
SEED_RUNS = {}
for k, fz in KEY.items():
    for sd in TR["seeds"]:
        name = f"{k.split(' ')[0]}{'_' + str(BEST_B) if k.startswith(('best', 'random')) else ''}_s{sd}"
        rf = 0.5 if fz == "replay" else 0.0
        SEED_RUNS[(k, sd)] = run_cached("12e_seeds", name, lambda fz=fz, rf=rf, sd=sd, name=name: train_cl(
            set() if fz == "replay" else fz, replay_frac=rf, seed=sd, tag=name))
if not is_done("12e_seeds"):
    mark_done("12e_seeds")
SEED_SUMMARY = {}
for k in KEY:
    rs = [SEED_RUNS[(k, sd)]["test"] for sd in TR["seeds"]]
    SEED_SUMMARY[k] = {m: {"mean": float(np.mean([r[m] for r in rs])), "sd": float(np.std([r[m] for r in rs]))}
                       for m in ("drift_forgetting_mean_pts", "forgetting_mean_pts", "new_word_acc", "new_word_auc", "hmean", "acc_13", "old_mean_acc")}
'''))
cells.append(cell("code", r'''# ---- CHECK 12e: mean +- SD over seeds
frozen_ok("12e", {f"{k}/{s}": v for (k, s), v in SEED_RUNS.items()})
if len(TR["seeds"]) > 1:
    check("12e different seeds give different runs", len({round(SEED_RUNS[("no_mask", s)]["test"]["drift_forgetting_mean_pts"], 6)
                                                        for s in TR["seeds"]}) > 1)
print(f"{'approach':<28}" + "".join(f"{m:>22}" for m in ("drift", "13-way forgetting", "zero acc", "hmean", "13-way acc")))
for k, s in SEED_SUMMARY.items():
    print(f"{k:<28}" + "".join(f"{s[m]['mean']:>14.3f} ± {s[m]['sd']:<5.3f}" for m in
                               ("drift_forgetting_mean_pts", "forgetting_mean_pts", "new_word_acc", "hmean", "acc_13")))
'''))

# ---------------------------------------------------------------- 12f
_ns = {}
exec(Path(__file__).with_name("stage12f_cells.py").read_text(encoding="utf-8"), _ns)
cells.append(cell("markdown", _ns["MD_12F"]))
cells.append(cell("code", _ns["CODE_12F"]))
cells.append(cell("code", _ns["CHK_12F"]))
cells.append(cell("markdown", _ns["MD_12G"]))
cells.append(cell("code", _ns["CODE_12G"]))
cells.append(cell("code", _ns["CHK_12G"]))

# ---------------------------------------------------------------- 13
cells.append(cell("markdown", '''## 13 · Summary

All key numbers in `RUN_DIR/summary.json`, with the list of failed checks; the figures are in `RUN_DIR/plots/`.'''))
cells.append(cell("code", r'''def slim(r):
    return {k: r[k] for k in ("val", "test", "frozen_per_layer", "weight_change", "settings", "method", "budget", "budget_frac",
                              "mask", "lr_mult", "replay_frac", "curve") if k in r}


hist = load_json(d7 / "history.json")
SUMMARY = {
    "run": load_json(RUN / "config.json"),
    "training": {"epochs_run": len(hist), "best_epoch": BASE_EVAL["best_epoch"],
                 "validation_acc": BASE_EVAL["validation"]["acc"], "test_acc": BASE_EVAL["testing"]["acc"],
                 "validation_loss": BASE_EVAL["validation"].get("loss"),
                 "test_per_class": dict(zip(CLASSES_12, BASE_EVAL["testing"]["per_class"])),
                 "history": [{k: r.get(k) for k in ("epoch", "loss", "train_acc", "val_loss", "val_acc", "lr")} for r in hist]},
    "fisher": load_json(d8 / "fisher_summary.json"),
    "mi": load_json(d9 / "mi_summary.json"),
    "groups": {**STATS, "mi_union_size": len(MI_UNION)},
    "ablation": {m: SPEC[m] for m in SPEC},
    "combo": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in COMBO.items()},
    "epochs": {"e_star": E_STAR, "mean_val_hmean": mean_val,
               "table": {str(e): {a: {"val": s["val"], "test": s["test"]} for a, s in row.items()} for e, row in EPOCH_TABLE.items()}},
    "arms": {**{k: slim(v) for k, v in ARM_RUNS.items()}, "slda_all": slim(SLDA_RUN)},
    "best_blend": {"name": BEST_BLEND, "mode": BEST_BLEND_MODE, "alpha": BEST_ALPHA},
    "budget": {k: slim(v) for k, v in BUDGET_RUNS.items()},
    "best_mask": {"name": BEST_MASK, "method": BEST_METHOD, "neurons": BEST_B},
    "tradeoff": {k: slim(v) for k, v in TRADE_RUNS.items()},
    "pareto_front": FRONT,
    "seeds": SEED_SUMMARY,
    "stability": STAB_SUMMARY,
    "stability_seeds": SEED12G_SUMMARY,
    "base_old_mean_acc": float(BASE_OLD.mean()),
    "checks": {"total": len(CHECKS), "failed": [k for k, v in CHECKS.items() if not v["passed"]],
               "skipped": [k for k, v in CHECKS.items() if v.get("skipped")]},
}
save_json(SUMMARY, RUN / "summary.json")
log("=" * 72)
log(f"summary written to {(RUN / 'summary.json').resolve()}")
log(f"base network: {len(hist)} epochs, best (lowest validation loss) at epoch {BASE_EVAL['best_epoch']} | "
    f"validation {BASE_EVAL['validation']['acc']:.4f} | test {BASE_EVAL['testing']['acc']:.4f}")
log(f"E* = {E_STAR} | best blend {BEST_BLEND} | best mask {BEST_MASK} ({BEST_B} neurons)")
log(f"checks: {len(CHECKS)} run, {len(SUMMARY['checks']['failed'])} failed, {len(SUMMARY['checks']['skipped'])} skipped"
    + (": " + "; ".join(SUMMARY["checks"]["failed"]) if SUMMARY["checks"]["failed"] else ""))
'''))

nb = {"cells": cells, "metadata": src["metadata"], "nbformat": src["nbformat"], "nbformat_minor": src["nbformat_minor"]}
out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", out_path, "with", len(cells), "cells")

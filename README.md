# Spiking keyword spotter: per-class neurons and learning a new word without forgetting

A delta-encoded spiking neural network (3 × 512 leaky integrate-and-fire neurons, trained with BPTT) recognises 12 classes of Google Speech Commands v2. The project asks:

1. **Which hidden neurons belong to which class?** Per-class mutual information (MI), Fisher information, Omega, and blends of MI and Fisher, tested by ablation.
2. **Can the network learn a 13th word ("zero") without forgetting the old ones?** Freezing chosen neurons, replay, distillation, soft freezing, output calibration and an SLDA head, compared under the same protocol.

Everything is plain PyTorch + NumPy (MFCC front end, encoder, LIF neurons and every analysis written from scratch). No internet is needed to run.

## Notebooks

| Notebook | What it does |
| --- | --- |
| `snn_gscv2_pipeline.ipynb` | The full pipeline: data, front end, encoder, base training (40 epochs), Fisher / Omega / MI, neuron groups, ablation, blended MI + Fisher groups (stage 11b), and 28 continual-learning arms including Forward-Forward. |
| `snn_gscv2_tradeoff.ipynb` | The forgetting–plasticity study: base training with early stopping on validation loss, α sweep of MI + Fisher blends, epoch sweep, freeze-size sweep, learning-rate × replay sweep, seeds, and a stability-first stage (distillation, calibration, soft freezing). |

Both notebooks run end to end with `SMOKE_TEST = True` on a tiny synthetic dataset (minutes), then on the real data with `SMOKE_TEST = False`. Every stage saves its outputs and is reloaded on a re-run; each stage has a check cell that records PASS/FAIL in `runs/<run>/checks.json`.

## Data

Not included. Download [Google Speech Commands v2](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz), extract it, and set `DATA_ROOT` in the notebook's config cell. The first real run decodes the audio once into an int16 cache (~1.5 GB, git-ignored).

## Results included

`runs/` holds every result except the audio cache and model weights:

| Folder | Run |
| --- | --- |
| `runs/gscv2_tradeoff/` | **Main real-data study** (`snn_gscv2_tradeoff.ipynb`): `summary.json`, `log.txt`, `checks.json`, plots, and one JSON per continual-learning run |
| `runs/gscv2_delta_3x512/` | Real-data run of the pipeline notebook (40-epoch base network) |
| `runs/smoke_test/`, `runs/smoke_test_base30_cl15/` | Synthetic smoke tests (6 and 30 base epochs) |

### Headline results (real GSC v2, test split)

Base network: **86.6%** test accuracy on 12 classes (early stopping at epoch 53). After learning "zero", mean ± SD over 3 seeds:

| Approach | Old-class accuracy | "zero" accuracy | 13-class accuracy |
| --- | --- | --- | --- |
| No protection | 60.4% | 97.4% | 63.5% |
| MI mask (322 neurons frozen) | 65.1% | 97.8% | 67.7% |
| Best MI + Fisher mask (768 frozen) | 69.9% | 98.2% | 72.2% |
| Random mask, same 768 neurons | 70.4% | 97.5% | 72.6% |
| 25% replay of labelled old clips | 80.6% | 97.0% | 81.9% |
| **Distillation (no old labels) + calibration** | **84.5%** | **80.0%** | **84.1%** |
| SLDA head on frozen spike counts (no training; deterministic, one run) | 85.8% | 87.1% | 85.9% |

- **MI identifies class-specific neurons:** ablation specificity +0.25, against +0.04 for Fisher.
- **Freezing MI- or Fisher-selected neurons protects no better than freezing a random set of the same size.** Selection only helps when 95% of the network is frozen.
- **Constraining the old outputs, by distillation or an SLDA head, keeps the old classes within about 2 points** of the base network.

## Scripts

- `scripts/run_nb.py`: runs every code cell of a notebook as one script (`--real` switches off the smoke test).
- `scripts/run_after_current.ps1`, `scripts/run_12f_after_current.ps1`: Windows helpers that wait for a running notebook and then resume the next stages (used with Task Scheduler).
- `tools/`: the generators that built `snn_gscv2_tradeoff.ipynb` from the pipeline notebook.

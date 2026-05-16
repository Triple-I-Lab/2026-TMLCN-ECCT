# SG-ECCT: Spectral-Guided Mask Generation for Error Correction Code Transformers

Official implementation of the paper:

> **Spectral-Guided Mask Generation for Error Correction Code Transformers**
> Linh Nguyen, Quoc Bao Phan, Tuy Tan Nguyen
> Department of Electrical and Computer Engineering, FAMU-FSU College of Engineering, Florida State University
> *IEEE Transactions on Machine Learning in Communications and Networking*

---

## Overview

SG-ECCT replaces the empirical cyclic-shift masks of MM-ECCT with principled
attention masks derived from spectral graph theory. The pipeline:

1. Build the variable-variable (V×V) graph from the parity-check matrix H
2. Compute the Fiedler vector of the normalised Laplacian
3. Identify the weakest structural cluster S_sw via conductance sweep
4. Enforce a scattered identity substructure at the weak columns (row ops only)
5. Use the resulting masks to guide the transformer's self-attention
---

## Installation

```bash
pip install -r requirements.txt
```

---

## Repository Structure

```
sg-ecct/
├── mask_generation.py   # Core: ConductanceBasedMaskGenerator
├── model.py             # SGECCT transformer model
├── code_generators.py   # BCH and Polar code generators
├── train.py             # Training script
├── evaluate.py          # BER/FER evaluation with SNR sweep
├── data/
│   └── codes/           # Pre-generated .npz code files (G, H matrices)
└── scripts/
    ├── prepare_codes.py # Generate and verify BCH + Polar codes
    ├── plot_fiedler.py  # Reproduce Figure 2 (Fiedler vector analysis)
    └── plot_results.py  # Reproduce BER figures and tables from paper
```

---

## Quick Start

### 1. Prepare codes
10 Codes are generated and verified, located in data/code/ folder
You can double-check using:
```bash
python scripts/prepare_codes.py --check
```

### 2. Train
Example: 
```bash
python train.py --code_name BCH_31_11 \
    --num_masks 2 --d_model 128 --num_heads 8 --num_layers 4 \
    --epochs 1000 --patience 30 \
    --train_samples 100000 --val_samples 10000
```

Checkpoints are saved to `experiments/<code_name>_<timestamp>/`.

### 3. Evaluate
Example: 
```bash
python evaluate.py \
    --checkpoint experiments/BCH_31_11_<timestamp>/best_model.pt \
    --snr_min 1 --snr_max 7 --snr_step 1 \
    --max_samples 50000 --min_fer_events 100
```

Results are saved as `eval_results.json` and `ber_curve.png` in the
checkpoint directory.

---

## Reproducing Paper Results

The paper reports BER for the following configurations:

| Code | h | d values |
|------|---|---------|
| BCH(31,11), BCH(31,16), BCH(63,36), BCH(63,51) | 4, 8 | 32, 64, 128, 256 |
| Polar(64,32), Polar(128,64), Polar(128,86) | 4, 8 | 32, 64, 128, 256 |
| LDPC(121,60), LDPC(144,72), LDPC(192,96) | 4, 8 | 32, 64, 128, 256 |

Train each configuration, collect results into `output_data.xlsx`,
then generate all figures and tables:

```bash
python scripts/plot_results.py           # all outputs
python scripts/plot_results.py --fig summary   # 6-panel BER grid
python scripts/plot_results.py --table ber     # BER @ SNR=7
```

To reproduce Figure 2 (Fiedler vector analysis):
```bash
python scripts/plot_fiedler.py --code BCH_31_11
```

---

## Running Tests

Each module has built-in tests:

```bash
python mask_generation.py
python model.py
python code_generators.py
python train.py --test
python evaluate.py --test
python scripts/prepare_codes.py --test
python scripts/plot_fiedler.py --test
python scripts/plot_results.py --test
```

---

## Citation

```bibtex
@article{nguyen2025sgecct,
  title   = {Spectral-Guided Mask Generation for Error Correction Code Transformers},
  author  = {Nguyen, Linh and Phan, Quoc Bao and Nguyen, Tuy Tan},
  journal = {IEEE Transactions on Machine Learning in Communications and Networking},
  year    = {2026},
}
```

---

## Acknowledgements

This research was supported by the National Science Foundation under
CISE/OAC Grant No. 2600417.

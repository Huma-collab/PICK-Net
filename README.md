# PICK-Net

Code for **"PICK-Net: A Physiology-Guided Input-Adaptive Deep Learning Framework for Interpretable Multi-Condition ECG Classification"**, submitted to *Artificial Intelligence in Medicine* (Elsevier).

PICK-Net is a 12-lead ECG classifier whose input-adaptive Gabor kernels bound frequency to the 0.5-40 Hz physiological range, combined with a Physiologically-Aware Co-occurrence Interaction Module (PACIM) for modeling MI-AVB co-occurrence. The paper reports a controlled, two-seed ablation isolating the model's frequency-bound kernel parameterization from its auxiliary physics-consistency losses, evaluated on PTB-XL with zero-shot cross-dataset generalization to CPSC-2018.

## Repository structure

- **`src/`** — Core model, training, ablation, baseline, and analysis scripts used to produce the results reported in the paper (model architecture, locked-protocol training, DeLong significance testing, kernel/clinical correlation analysis, cross-dataset evaluation, and figure generation).
- **`pathA_project/`** — Self-contained code for the locked-protocol, four-condition ablation campaign (Full model, A2, Rotating-Subset, Unbounded) that isolates the frequency bound's contribution from the physics-consistency losses (Section 5.9 of the paper). See its own `README.md` for details.

## Data

This work uses two publicly available ECG datasets:
- [PTB-XL](https://physionet.org/content/ptb-xl/) (Wagner et al., 2020)
- [CPSC-2018](http://2018.icbeb.org/Challenge.html) (China Physiological Signal Challenge)

Neither dataset is included in this repository; both must be obtained separately from their original sources.

## License

MIT License — see [LICENSE](LICENSE).

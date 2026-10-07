# Evaluating Large Language Models in High-Stakes Domains — companion repository

Code, simulated-data generators, de-identified study data and figures for the monograph
*Evaluating Large Language Models in High-Stakes Domains: A Gentle Introduction to Measurement, Uncertainty, and Reliability*
(Thalles Ferreira Costa, Oswaldo Cruz Institute, IOC/Fiocruz; draft prepared for *Foundations and Trends® in Machine Learning*).

## Contents

| Path | What it is |
|---|---|
| `figures.py` | Generates every simulated figure and every number quoted in the worked examples (Chapters 2–6). Writes `figures/*.png` and `figures/results.json`. |
| `realdata.py` | Analysis for Section 6.7 (real study). Reads the study's private audit package; the de-identified output it produces is in `data/`. |
| `data/study_percall_deidentified.csv` | Per-call records of the fifteen-model transfusion-medicine study (14 Sept 2026): model (A–O), condition, item, tokens, reasoning tokens, upstream cost, answer length, finish reason. Model identities and vignette text are withheld until the study's primary publication. |
| `notebooks/ch2.ipynb` … `ch6.ipynb` | One notebook per chapter, generated from `figures.py`, runnable top to bottom. |
| `notebooks/ch6_7_real_study.ipynb` | Reproduces Figure 7 and Table 6.5 from the de-identified CSV with pandas. |
| `figures/` | Rendered figures and the JSON of quoted numbers. |

## Reproducing

```bash
pip install -r requirements.txt
python figures.py            # ~2 minutes; writes figures/ and results.json
jupyter lab notebooks/       # or run any notebook top to bottom
```

All simulations are seeded; the numbers in the monograph are those in `figures/results.json` and `figures/results_real.json`.

## Citation

Costa, T. F. (2026). *Evaluating Large Language Models in High-Stakes Domains: companion code, data and notebooks* (v0.1.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.23216193

The Zenodo record archives release `v0.1.0` of this repository.

## Releasing a new version

Tag the commit, push the tag, and publish a new Zenodo version with the script in `tools/` (needs a Zenodo personal token in `ZENODO_TOKEN`):

```bash
git tag -a v0.2.0 -m "..." && git push --tags
python tools/zenodo_new_version.py v0.2.0
```

The script archives the tag with `git archive`, opens a new version of the Zenodo record, replaces the file, sets the version and publishes; the concept DOI stays the same and a version DOI is minted.

## License

MIT for code (see `LICENSE`). The de-identified data are released under CC BY 4.0.

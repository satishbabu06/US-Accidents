# US-Accidents

# US Accidents — Severity Prediction

**Ironhack DSML Final Project**

Predicting whether a US traffic accident will have a **severe impact on traffic flow** (Severity 3 or 4) based on conditions known at the *start* of the incident — location, time, weather, and road features.

> **Note on "severity":** in this dataset, severity measures *impact on traffic / delay duration*, NOT injury severity.

## Dataset

- Source: [US Accidents (Kaggle, Sobhan Moosavi)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
- ~7.7 M rows, 47 columns, Feb 2016 – Mar 2023, ~3 GB CSV
- Download and place `US_Accidents_March23.csv` in the `data/` folder.

## Project Structure

```
final-project/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py                          ← Streamlit demo
├── data/                           ← raw + cleaned data (gitignored)
│   └── US_Accidents_March23.csv    (you download this)
├── notebooks/
│   ├── 01_eda.ipynb                ← fast loading + EDA + stat tests
│   └── 02_modeling.ipynb           ← ML + DL + tuning + final model
├── models/                         ← saved models (gitignored)
└── reports/figures/                ← generated plots
```

## How to Run

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate                  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Download US_Accidents_March23.csv from Kaggle → place in data/

# 3. Run the notebooks in order
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_modeling.ipynb

# 4. (Optional) Launch the demo
streamlit run app.py
```

## Why this project

The brief asks for ML, DL, and computer vision options. This project covers ML + DL via a multi-model comparison on a real-world tabular dataset with substantial class imbalance, target leakage traps, and a meaningful business framing (traffic dispatch).

## Key Decisions

| Decision | Choice | Reasoning |
|---|---|---|
| Loading strategy | `pyarrow` engine + Parquet cache | Default `pd.read_csv` takes ~1 hour on 3 GB; this approach is ~5 seconds after first load |
| Target framing | Binary (severe ≥3 vs not) | Imbalance + clearer business question |
| Sample size | 500 K stratified | Fast iteration; final model retrained on full set |
| Leakage columns dropped | `End_Time`, `End_Lat`, `End_Lng`, `Distance(mi)`, `Description` | Only known *after* the accident |
| Imbalance handling | `class_weight='balanced'` + threshold tuning | Faster than SMOTE, comparable results |
| Primary metric | Macro-F1 + PR-AUC | Accuracy is misleading on 80/20 imbalance |
| Final model | LightGBM | Best macro-F1, fast, interpretable |

## EDA Highlights

The EDA notebook covers:
- Target distribution (4-class + binarized)
- Time patterns (hour, day-of-week, month, yearly trend incl. COVID)
- Geographic patterns (state, lat/lng scatter on US map)
- Weather analysis
- Road feature lift
- Numeric distributions by severity (violin plots)
- Correlation heatmap
- **Statistical tests**: Mann-Whitney U for numerics, Chi-square + Cramér's V for categoricals

## Modeling Approach

| Model | Role |
|---|---|
| Dummy | Floor — confirms features have signal |
| Logistic Regression | Linear baseline |
| Random Forest | Non-linear, captures interactions |
| LightGBM | State-of-the-art for tabular |
| Keras MLP | Deep learning comparison (3 hidden layers) |

After model comparison, LightGBM was tuned with RandomizedSearchCV (20 iterations, 3-fold CV on macro-F1) and the decision threshold was tuned to hit 80% recall on the severe class.

## Results

Typical numbers on a 500 K stratified sample (exact values depend on the sample):

| Model | Macro F1 | PR-AUC | Training time |
|---|---|---|---|
| Dummy | ~0.45 | ~0.20 | <1s |
| LogReg | ~0.55–0.60 | ~0.30 | ~30s |
| Random Forest | ~0.65–0.70 | ~0.40 | ~2 min |
| **LightGBM (tuned)** | **~0.70–0.75** | **~0.45+** | ~3 min |
| MLP | ~0.65–0.70 | ~0.40 | ~5 min |

After threshold tuning: ~80% recall, ~50–60% precision on the severe class.

## Challenges Faced

- **Loading time** — default `pd.read_csv` took ~1 hour. Solved with pyarrow + Parquet caching.
- **Target leakage** — `Distance(mi)` is the worst offender; identified and removed before any modeling.
- **Class imbalance** — switched to macro-F1 / PR-AUC, used `class_weight='balanced'`, tuned threshold.
- **High-cardinality categoricals** — collapsed 100+ weather strings into ~10 groups; one-hot with `min_frequency=50` for State.
- **Tabular DL underperformance** — expected; documented honestly rather than pretending the MLP won.

## Limitations

- Dataset biased toward CA/FL/TX → poor generalization to other states is plausible.
- "Severity" = traffic-flow impact, NOT injury severity.
- Predicted probabilities not calibrated after `class_weight` rebalancing.

## License

Educational / academic use. Dataset © Sobhan Moosavi (CC BY-NC-SA 4.0).

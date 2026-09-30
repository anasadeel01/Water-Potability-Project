# 💧 Water Quality Risk Classification
**Learn Depth Academy LLP — Track 1 Final Capstone, Problem 09**

Classifies a water sample as **potable (safe)** or **not potable (unsafe)** using nine measurable physicochemical properties, with a Streamlit demo app for live predictions.

## Problem
| | |
|---|---|
| Domain | Environment |
| ML Task | Binary Classification |
| Duration | 10 days |
| Model constraint | Foundational methods only (Logistic Regression, KNN, Decision Tree) |

## Dataset
- **Source**: [Kaggle — Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability) (public, open, free)
- 3,276 samples, 9 numeric features, binary target `Potability`
- Features: `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`
- License: allows non-commercial/educational reuse — cited here per capstone rules, not redistributed for any other purpose.

## Project structure
```
├── data/
│   └── water_potability.csv          # raw dataset (cited above)
├── notebooks/
│   └── water_potability_capstone.ipynb   # full EDA + modelling notebook, pre-executed
├── src/
│   └── train.py                       # reproducible training script (same pipeline as notebook)
├── models/
│   ├── best_model.pkl                 # trained Decision Tree
│   ├── scaler.pkl                     # StandardScaler (fit on train split only)
│   ├── imputer.pkl                    # SimpleImputer (median, fit on train split only)
│   └── feature_order.json             # exact feature column order the model expects
├── assets/                            # EDA + evaluation plots (png)
├── reports/
│   └── metrics_summary.json           # all model metrics in one file
├── app/
│   └── app.py                         # Streamlit prototype
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

# retrain from scratch (optional — models/ already contains a trained model)
python src/train.py

# launch the demo app
streamlit run app/app.py
```

Or open `notebooks/water_potability_capstone.ipynb` in Jupyter to see the full EDA and model-comparison walkthrough (already executed with outputs, so it can be read without running anything).

## Approach
1. **EDA**: checked missing values (`ph`, `Sulfate`, `Trihalomethanes`), duplicates (none), class balance (~61/39), feature distributions by class, and correlation with the target.
2. **Preprocessing**: median imputation + standard scaling, both fit **only on the training split** to avoid data leakage.
3. **Modelling**: trained Logistic Regression, KNN, and a Decision Tree (the three foundational methods permitted by the brief), with 5-fold stratified cross-validation.
4. **Evaluation**: accuracy, precision, recall, F1, ROC-AUC, and confusion matrices for all three; selected the Decision Tree.
5. **Prototype**: a Streamlit app that takes the same 9 inputs, applies the identical imputation + scaling pipeline, and returns a prediction with a probability estimate.

## Results summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.52 | 0.42 | 0.53 | 0.47 | 0.55 |
| KNN (k=15) | 0.62 | 0.52 | 0.24 | 0.33 | 0.61 |
| **Decision Tree (selected)** | **0.64** | **0.60** | **0.23** | 0.33 | **0.61** |

*(Precision/Recall/F1 above are for the "Potable" (1) class; see `reports/metrics_summary.json` and the technical paper for both classes.)*

## Honest limitation (important)
This is a widely studied public dataset, and it is well known in the ML community that **none of its 9 features correlate strongly with `Potability`** (all |r| < 0.05). That means even well-tuned foundational models plateau around 60–65% accuracy — this is a property of the data, not a bug in the pipeline. The Decision Tree was selected because it has the best recall on the safety-critical "Not Potable" class (catches ~90% of unsafe samples), even though its overall F1 is modest. Full discussion is in the technical paper's Limitations section.

## Credits
- Dataset: Kaggle, "Water Potability" by Aditya Kadiwal (public domain / open dataset).
- Built with: pandas, scikit-learn, seaborn/matplotlib, Streamlit.
- Capstone rule compliance: free/open-source tools only, no paid APIs.

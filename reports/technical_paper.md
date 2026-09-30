# Water Quality Risk Classification: A Foundational Machine Learning Approach

**Track 1 Final Capstone — Problem 09 | Learn Depth Academy LLP**

---

## Abstract
Access to safe drinking water is a basic public health requirement, yet water testing infrastructure is not universally available. This project investigates whether a lightweight, zero-cost machine learning model can classify a water sample as potable (safe) or not potable (unsafe) from nine routinely measurable physicochemical properties. Using the public Kaggle "Water Potability" dataset (3,276 samples), we compare three foundational classifiers — Logistic Regression, K-Nearest Neighbours, and a Decision Tree — under a leakage-safe preprocessing pipeline. The Decision Tree is selected as the final model, achieving 64% test accuracy and a ROC-AUC of 0.61, with 90% recall on the safety-critical "not potable" class. We find that none of the nine features correlate strongly with potability, which caps achievable performance for any foundational model on this dataset; this limitation is analysed explicitly rather than concealed behind a single headline metric. A Streamlit prototype demonstrates the end-to-end pipeline on new inputs.

## 1. Introduction
Manual water-quality testing (lab assays for pH, hardness, dissolved solids, etc.) is slow and resource-intensive, especially in low-resource settings. A model that flags likely-unsafe samples from a handful of low-cost sensor or lab readings could help prioritise which samples need urgent, more expensive confirmatory testing. This project builds such a model end-to-end — from raw data to a usable demo — within a 10-day, zero-cost constraint, using only foundational (non-deep-learning) methods as required by the capstone brief.

## 2. Problem Definition
**Task**: Binary classification — given nine physicochemical measurements of a water sample, predict whether it is `Potable` (1, safe) or `Not Potable` (0, unsafe).
**Constraints**: free/open-source tools only; no paid APIs; foundational ML methods only (Logistic Regression, KNN, Decision Tree); solution must be reproducible and explainable; 10-day timeline.

## 3. Related Work
The Kaggle "Water Potability" dataset is one of the most widely used public benchmarks for water-safety classification tutorials and student projects. Prior public analyses (Kaggle notebooks, GitHub student projects, and applied-ML blog posts) consistently report that tree-based and instance-based foundational models reach roughly 60–68% accuracy on this dataset, and that feature-target correlations are uniformly weak. This project's results are consistent with that broader pattern, and the same limitation is discussed below rather than treated as a modelling failure.

## 4. Dataset
- **Source**: Kaggle, "Water Potability" dataset (public, open license), 3,276 rows.
- **Features** (9, all numeric): `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`.
- **Target**: `Potability` — binary (0 = not potable, 1 = potable).
- **Class balance**: 1,998 not-potable (61%) vs 1,278 potable (39%) — moderate imbalance.
- **Missing values**: `ph` (491, ~15%), `Sulfate` (781, ~24%), `Trihalomethanes` (162, ~5%). No duplicate rows.

## 5. Methodology
1. **Data cleaning**: verified no duplicate records; confirmed the three columns with missing values above.
2. **Train/test split**: 80/20 stratified split (random_state=42) so the class ratio is preserved in both sets.
3. **Preprocessing** (fit on the training split only, to avoid data leakage):
   - Median imputation for the three columns with missing values.
   - Standard (z-score) scaling of all nine features.
4. **Models trained** (foundational methods only, per capstone scope):
   - Logistic Regression (`class_weight="balanced"`, max_iter=1000)
   - K-Nearest Neighbours (k=15)
   - Decision Tree (`max_depth=6`, `class_weight="balanced"`)
5. **Validation**: 5-fold stratified cross-validation (mean F1) on the training split, plus a held-out test-set evaluation.
6. **Model selection criterion**: ROC-AUC as the primary metric (robust to the class imbalance), with recall on the "not potable" class as a practical tie-breaker, because in a water-safety context a false negative (calling unsafe water "potable") is the costlier error — not raw accuracy alone.

## 6. Exploratory Data Analysis — Key Findings
- **Class balance**: moderately imbalanced (61/39); addressed via `class_weight="balanced"` rather than resampling, keeping the pipeline simple and reproducible.
- **Missing values**: concentrated in three features, consistent with plausible real-world sensor/lab gaps; median imputation was chosen over deletion to avoid losing ~30% of rows (a large share are missing `Sulfate`).
- **Feature distributions by class**: visually near-identical between the potable and non-potable classes for all nine features (see `assets/feature_distributions.png`) — an early warning sign of weak separability.
- **Correlation with target**: every feature shows |Pearson r| < 0.05 with `Potability` (see `assets/correlation_heatmap.png`). This is the single most important EDA finding, because it directly predicts the modest ceiling on achievable model performance discussed in Section 8.
- **No obvious data leakage risk**: none of the nine features are derived from or trivially encode the target.

## 7. Model Development and Results

| Model | Accuracy | Precision (1) | Recall (1) | F1 (1) | ROC-AUC | CV F1 (mean) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.524 | 0.415 | 0.531 | 0.466 | 0.547 | 0.417 |
| KNN (k=15) | 0.617 | 0.521 | 0.238 | 0.327 | 0.605 | 0.371 |
| **Decision Tree (selected)** | **0.639** | **0.596** | 0.230 | 0.332 | **0.607** | 0.353 |

For the selected Decision Tree, class-wise behaviour on the held-out test set:
- **"Not Potable" (0)** — precision 0.65, **recall 0.90**, F1 0.75
- **"Potable" (1)** — precision 0.60, recall 0.23, F1 0.33

Confusion matrices and ROC curves for all three models are saved under `assets/`.

## 8. Discussion
The Decision Tree was chosen primarily for its high recall (0.90) on the "not potable" class: in this deployment context, missing an unsafe sample is a more serious error than over-flagging a safe one for re-testing. Its overall accuracy (64%) and ROC-AUC (0.61) also modestly exceed the other two foundational models. This comes at a clear cost — recall on the "potable" class is only 0.23, meaning the model is conservative and would send many genuinely safe samples for unnecessary re-testing. Logistic Regression, by contrast, is more balanced across classes (F1 0.47 for the potable class) but has the lowest ROC-AUC of the three. KNN sits between the two. None of the three approaches is dramatically superior, which is expected given the correlation analysis in Section 6.

## 9. Limitations
1. **Weak feature–target relationship**: as shown in the correlation heatmap, none of the nine features are strongly predictive of potability in isolation or in simple combination — this is a property of the public dataset, not of the modelling choices, and it caps performance for any foundational (non-deep, non-heavily-engineered) approach at roughly 60–65% accuracy.
2. **No domain feature engineering**: no interaction terms, WHO/EPA threshold-based binary flags, or external water-quality-index features were engineered, per the capstone's "foundational methods" scope; these could plausibly improve performance but were out of scope for a 10-day foundational-model project.
3. **Imputation assumption**: median imputation assumes missingness is not strongly related to the target (MCAR/MAR-like); this was not formally tested (e.g. via a missingness indicator model) due to time constraints.
4. **Not a certified safety tool**: this prototype is an educational exercise and has not been validated against any regulatory drinking-water standard (e.g. WHO or EPA guidelines) or on any real-world sample outside this dataset.
5. **Static dataset size**: 3,276 samples is modest for a health-safety classification task; results may not generalise to water sources or contamination profiles not represented in the source data.

## 10. Future Scope
- Try ensemble foundational methods (Random Forest, Gradient Boosting) if the brief's scope is extended beyond single foundational classifiers.
- Engineer domain-informed features (e.g., distance from WHO safe thresholds per parameter) rather than relying on raw values alone.
- Collect or merge additional real-world labelled data to test generalisation beyond this single public dataset.
- Add model explainability (e.g. decision-path visualisation or feature importance) directly into the Streamlit app for end-user trust.

## 11. Conclusion
This project delivers a complete, reproducible, zero-cost pipeline — from raw data to a working Streamlit demo — for water potability classification using only foundational ML methods, as required by the capstone brief. The selected Decision Tree model reaches a modest but honestly reported 64% accuracy and 0.61 ROC-AUC, with strong recall on the safety-critical unsafe-water class. The central, well-supported finding is that this ceiling reflects the dataset's weak feature–target relationships rather than a flaw in the modelling pipeline — a distinction we consider more valuable to report clearly than to obscure with metric selection.

## 12. References
1. Kadiwal, A. *Water Potability* [Dataset]. Kaggle. https://www.kaggle.com/datasets/adityakadiwal/water-potability
2. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
3. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*.
4. Waskom, M. (2021). seaborn: statistical data visualization. *Journal of Open Source Software*, 6(60), 3021.
5. Streamlit Inc. Streamlit documentation. https://docs.streamlit.io

---
*Prepared as part of the Learn Depth Academy LLP Track 1 Final Capstone submission. All tools used are free/open-source, per the capstone's scope and constraints.*

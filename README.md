# 💡 Key Insights

This document concisely summarizes the key insights earned during the Porto Seguro’s Safe Driver Prediction.

## 🎯 Lessons Learned During Problem Solving

* **Deep Understanding of Missing Values**: Missing values in datasets are not always represented by `NaN`; they can be encoded with specific numbers like `-1`.
* **Informative Missing Data**: Missing values are often not just absent data but can provide meaningful information to the predictive model itself.
* **Careful Feature Dropping**: Before dropping a feature, always check its distribution to avoid distortion from sparsity or low sample counts.
* **Evaluation Metric-Centric Learning**: It's crucial to evaluate the model and monitor the learning process according to the main evaluation metric of the competition (e.g., Gini Coefficient) for optimal final performance. `lgb.log_evaluation` callback helps monitor training progress, and `lgb.early_stopping` callback prevents overfitting.
* **Consistent Parameter Management**: In LightGBM training, it's best practice to keep core model hyperparameters within the `params` dictionary and pass execution-related control arguments (`num_boost_round`, `valid_sets`, `callbacks`, etc.) directly to the `lgb.train()` function. This improves code clarity and maintainability.

## 🛠️ Key Technologies Used

* Python
* Pandas
* NumPy
* LightGBM
* XGBoost
* Matplotlib
* Scikit-learn

---

1. Missing values are not always NaN. Can be other specific numbers like -1.
2. Missing values are often informative, not just absent datas
3. Before dropping a feature, always check its  distribution to avoid distortion from sparsity or low sample counts

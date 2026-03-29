# Vehicle Insurance Fraud Detection

Built a model to flag fraudulent vehicle insurance claims before they get paid out. The idea is straightforward - every fraud claim that slips through costs ~$15K on average, but investigating a flagged claim only costs ~$250, so even a decent model saves a lot of money.

## What this project does

Uses XGBoost on the [Kaggle Vehicle Claim Fraud Detection dataset](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) (~15K claims) to predict which claims are fraudulent. The model outputs a probability score per claim, and I pick a threshold (0.7) that balances catching fraud against not drowning the investigations team in false alarms.

## Results

At threshold 0.7:
- **ROC AUC: 0.882**
- Catches **55.2%** of fraud, **8.3%** false alarm rate
- Estimated **$2.17M in annual savings** versus just paying out every claim

55% detection might sound low, but at this threshold the model is quite precise about what it does flag - and each caught case saves $15K. The cost math works out heavily in favor of deploying it.

Top features were policy type (sedan liability), fault party, and base policy type. Basically: liability-heavy policies where a third party is at fault are the biggest fraud signal, which makes sense if you think about how staged accident schemes work.

## How I got here

Started by trying Random Forest, Logistic Regression, Gradient Boosting, and XGBoost. XGBoost won on AUC.

The class imbalance (~6% fraud) was the main challenge. I initially used SMOTE to oversample the minority class, but it was overfitting pretty badly. Switched to XGBoost's built-in `scale_pos_weight` parameter instead, which gave much cleaner generalization - train/test AUC gap came down to 0.063.

Tuned hyperparameters with GridSearchCV over tree depth, regularization strength, learning rate, and subsampling. Added fairly aggressive regularization (`reg_alpha=2`, `reg_lambda=2`, `min_child_weight=20`) to keep the model honest.

For the threshold, I evaluated 0.1 through 0.5 on recall vs. false alarm rate, then built a cost model to pick the sweet spot. 0.7 minimizes total expected cost given the $15K/$250 asymmetry.

## Visuals

The `outputs/` folder has:
- ROC curve with the 0.7 threshold marked
- Confusion matrix with counts and percentages
- Cost comparison bar chart (with vs. without model)
- Top 10 feature importances

## Running it

```bash
git clone https://github.com/[your-username]/insurance-fraud-detection.git
cd insurance-fraud-detection
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb
```

Needs Python 3.x, scikit-learn, XGBoost, imbalanced-learn, pandas, numpy, matplotlib, seaborn, kagglehub. The dataset downloads automatically via kagglehub when you run the notebook.

## Things I'd improve

- SHAP values instead of just feature importances - importances don't capture interaction effects
- Calibration curve to check if the predicted probabilities are actually reliable at the extremes
- Sensitivity analysis on the cost assumptions - how much does the savings estimate move if average fraud is $12K or $20K?
- Temporal validation split to check for distribution drift
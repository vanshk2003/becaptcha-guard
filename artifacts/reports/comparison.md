# Model comparison (held-out test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | EER |
|---|---|---|---|---|---|---|
| logreg | 0.891 | 0.955 | 0.828 | 0.887 | 0.955 | 0.127 |
| svm_rbf | 0.861 | 0.952 | 0.770 | 0.851 | 0.929 | 0.163 |
| random_forest | 0.881 | 0.985 | 0.781 | 0.871 | 0.939 | 0.137 |
| xgboost | 0.887 | 0.935 | 0.840 | 0.885 | 0.957 | 0.117 |
| deep_behaviornet | 0.960 | 0.931 | 0.996 | 0.962 | 0.997 | 0.020 |

## Per-family recall (did we catch each bot type?)

| Model | bezier_bot_recall | human_TNR | minjerk_adv_recall | puppeteer_step_recall | selenium_linear_recall |
|---|---|---|---|---|---|
| logreg | 0.984 | 0.958 | 0.328 | 1.000 | 1.000 |
| svm_rbf | 1.000 | 0.958 | 0.078 | 1.000 | 1.000 |
| random_forest | 1.000 | 0.988 | 0.125 | 1.000 | 1.000 |
| xgboost | 1.000 | 0.938 | 0.359 | 1.000 | 1.000 |
| deep_behaviornet | 1.000 | 0.921 | 0.984 | 1.000 | 1.000 |
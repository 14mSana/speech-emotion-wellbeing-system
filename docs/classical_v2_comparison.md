# v2 Classical-Model Comparison

## Protocol

This local experiment uses `acoustic_v2_40_spectral`: the immutable 32-value
baseline vector plus spectral centroid, bandwidth, 85% rolloff, and spectral
contrast mean/std pairs. The RAVDESS actor split was fixed before fitting:

| Partition | Actors | Recordings |
| --- | --- | ---: |
| Train | 01–16 | 576 |
| Validation | 17–20 | 144 |
| Test | 21–24 | 144 |

All feature extraction uses the persisted shared audio configuration (22,050
Hz mono, silence trimming at 30 dB). There is no dataset-fitted acoustic
normalization. Logistic Regression uses a `StandardScaler` inside its pipeline;
the scaler was fit only on the training rows. XGBoost uses its tree features
directly. Candidate selection used validation macro F1 only; test data was not
materialized until both configurations had been selected.

The local, ignored artifacts are `ravdess_v2_logistic_regression.joblib`,
`ravdess_v2_xgboost.joblib`, `ravdess_v2_random_forest.joblib`,
`ravdess_v2_rbf_svm.joblib`, their metadata JSON files, and the v2 comparison
JSON. They are separate from the production Random Forest and the earlier
v1 RBF-SVM experiment.

## Validation selection

| Model | Candidate grid | Selected configuration | Validation accuracy | Validation macro precision | Validation macro recall | Validation macro F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Logistic Regression | `C` = 0.1, 1, 10; balanced classes | `C=0.1` | 0.5694 | 0.5535 | 0.5500 | 0.5459 |
| XGBoost | depth/learning-rate = (3, .05), (4, .05), (3, .10) | depth=4, learning_rate=0.05, 200 trees | 0.5417 | 0.5126 | 0.5000 | 0.4971 |
| Random Forest | 300 trees; `max_features` = `sqrt` or `None` | `max_features="sqrt"` | 0.5208 | 0.4923 | 0.4938 | 0.4888 |
| RBF-SVM | `C` = 0.1, 1, 10; `gamma="scale"`; balanced classes | `C=1.0` | 0.5625 | 0.5658 | 0.5750 | 0.5515 |

All model selection runs use `random_state=42`. XGBoost also uses
`subsample=0.8`, `colsample_bytree=0.8`, `tree_method="hist"`, and one worker
for reproducibility. The v2 Random Forest uses balanced class weights and 300
trees. The RBF-SVM's `StandardScaler` is fitted inside the training-only
pipeline.

## One-time held-out test results

The following test metrics were recorded exactly once after validation
selection. The label and confusion-matrix order is **Neutral, Calm, Happy,
Sad, Angry**; rows are true labels and columns are predictions.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression (v2) | 0.6250 | 0.5783 | 0.5875 | 0.5755 |
| XGBoost (v2) | 0.6181 | 0.6050 | 0.5688 | 0.5591 |
| Random Forest (v2) | 0.6042 | 0.5695 | 0.5563 | 0.5450 |
| RBF-SVM (v2) | 0.6042 | 0.5702 | 0.5813 | 0.5671 |

### Logistic Regression (v2)

| Emotion | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Neutral | 0.3636 | 0.2500 | 0.2963 |
| Calm | 0.5833 | 0.6563 | 0.6176 |
| Happy | 0.6667 | 0.8125 | 0.7324 |
| Sad | 0.5000 | 0.3438 | 0.4074 |
| Angry | 0.7778 | 0.8750 | 0.8235 |

```text
[[ 4,  4,  4,  3,  1],
 [ 4, 21,  0,  5,  2],
 [ 1,  2, 26,  3,  0],
 [ 2,  9,  5, 11,  5],
 [ 0,  0,  4,  0, 28]]
```

### XGBoost (v2)

| Emotion | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Neutral | 0.5000 | 0.1250 | 0.2000 |
| Calm | 0.6429 | 0.5625 | 0.6000 |
| Happy | 0.5319 | 0.7813 | 0.6329 |
| Sad | 0.5625 | 0.5625 | 0.5625 |
| Angry | 0.7879 | 0.8125 | 0.8000 |

```text
[[ 2,  5,  7,  2,  0],
 [ 1, 18,  2,  9,  2],
 [ 1,  0, 25,  3,  3],
 [ 0,  5,  7, 18,  2],
 [ 0,  0,  6,  0, 26]]
```

### Random Forest (v2)

| Emotion | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Neutral | 0.4000 | 0.1250 | 0.1905 |
| Calm | 0.5714 | 0.6250 | 0.5970 |
| Happy | 0.5581 | 0.7500 | 0.6400 |
| Sad | 0.5000 | 0.4375 | 0.4667 |
| Angry | 0.8182 | 0.8438 | 0.8308 |

```text
[[ 2,  5,  7,  2,  0],
 [ 0, 20,  1, 10,  1],
 [ 2,  2, 24,  2,  2],
 [ 1,  8,  6, 14,  3],
 [ 0,  0,  5,  0, 27]]
```

### RBF-SVM (v2)

| Emotion | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Neutral | 0.2727 | 0.3750 | 0.3158 |
| Calm | 0.6061 | 0.6250 | 0.6154 |
| Happy | 0.6486 | 0.7500 | 0.6957 |
| Sad | 0.5000 | 0.2813 | 0.3600 |
| Angry | 0.8235 | 0.8750 | 0.8485 |

```text
[[ 6,  4,  4,  2,  0],
 [ 7, 20,  0,  5,  0],
 [ 4,  2, 24,  2,  0],
 [ 5,  7,  5,  9,  6],
 [ 0,  0,  4,  0, 28]]
```

## Interpretation

Logistic Regression has the highest macro F1 of the four v2 experiments. The
v2 RBF-SVM is second and has the highest angry F1, while v2 XGBoost has the
highest macro precision but substantially lower neutral recall. All four still
show the established difficult boundaries around neutral/calm/sad; none is a
replacement for the production model. The historical production Random Forest
result was 0.6181 accuracy and 0.5794 macro F1 on the same actor test split,
but it used the 32-feature v1 schema. That historical result is useful context,
not a like-for-like v2 model ranking. The earlier RBF-SVM is likewise a v1
artifact and was not modified or re-evaluated here.

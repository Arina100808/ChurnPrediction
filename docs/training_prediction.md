# Model Training and Prediction Process

The core model used in this solution is the **XGBoost classifier**.

## Preprocessing Pipeline

A scikit-learn `Pipeline` is used to chain preprocessing and model training steps, including:

- **One-Hot Encoding**: Categorical variables are transformed using `OneHotEncoder` with `handle_unknown="ignore"` to ensure that unseen categories during prediction do not break the pipeline. 
- **Numerical Features**: Numerical columns are passed through without scaling. Since XGBoost is a tree-based model, feature scaling, such as standardization or normalization, is not required or beneficial.

## Training

The training logic is implemented in the `cmd_train` function:

1. Load the dataset from a CSV file.
2. If a market is specified, filter the data for that market using the column name `country` by default.
   
   _If a specific market is selected, the model is both trained and evaluated on data from that market only, and the market-identifying column (e.g., `country`) is dropped as it becomes redundant. However, if all markets are included, the `country` column is treated as a categorical feature during training._
3. Optionally load model configuration from a YAML file (one per market).
4. Split the data into training and testing sets (80/20).
5. Train the pipeline on the training data.
6. Evaluate performance on the test set using accuracy and AUC.
7. Save the following outputs:

### Saved Artifacts

Artifacts are saved under the `artifacts/{market}` folder:

| File                      | Format      | Description                                      |
|---------------------------|-------------|--------------------------------------------------|
| `model_{market}.joblib`   | Joblib      | Trained scikit-learn pipeline with XGBoost       |
| `metrics.json`            | JSON        | Evaluation metrics on the test set               |
| `params.json`             | JSON        | Model hyperparameters used during training       |
| `pred_{market}.csv`       | CSV         | Predicted probabilities on the test set          |

## Prediction

Prediction is handled by the `cmd_predict` function and follows these steps:

1. Load a previously trained model (must match the target market).
2. Load a dataset to score.
3. Drop the target column if present.
4. Optionally filter by market (same as during training).
5. Predict churn probabilities using the pipeline.
6. Save predictions to `scored_{market}.csv`.

**Note:** In this solution, models can only be used to predict data for the **same market** they were trained on. Cross-market inference is not supported.

## Market Configuration

Each market can have its own configuration file stored under `src/config/markets/{market}.yaml`. These YAML files can specify:

- `features`: List of selected features to include.
- `threshold`: Custom classification threshold for positive class.
- `model_params`: Dictionary of parameters to override default XGBoost settings.

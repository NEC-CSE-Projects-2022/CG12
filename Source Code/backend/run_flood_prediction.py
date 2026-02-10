"""run_flood_prediction.py

Small runnable script (CLI) to preprocess the flood dataset, train a RandomForest
classifier and save results. Defaults to reading
`backend/datasets/floodpredictiondataset.csv`.

Usage:
  python backend/run_flood_prediction.py --csv backend/datasets/floodpredictiondataset.csv

Output:
  - backend/output/predictions.csv  (index, prob, pred)
  - backend/output/rf_model.pkl     (saved model)

This implements a safe subset of the Colab notebook you provided.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error, confusion_matrix
import joblib


def main(csv_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}. Please place your dataset there or pass --csv path.")

    df = pd.read_csv(csv_path)
    print("Loaded CSV:", csv_path, "-> shape:", df.shape)

    # Basic cleaning
    df.columns = df.columns.str.strip()

    # If Rainfall missing entirely, create synthetic values between 50 and 300
    if 'Rainfall' not in df.columns:
        df['Rainfall'] = np.random.uniform(50, 300, size=len(df))
        print("Note: 'Rainfall' column missing — filled with synthetic values (50-300).")
    else:
        df['Rainfall'] = df['Rainfall'].fillna(df['Rainfall'].mean())

    # Numeric columns suggested by the notebook. We'll only keep those present in df.
    numeric_cols = ['MonsoonIntensity', 'TopographyDrainage', 'RiverManagement',
                    'Deforestation', 'Urbanization', 'ClimateChange', 'DamsQuality',
                    'Siltation', 'AgriculturalPractices', 'Encroachments',
                    'IneffectiveDisasterPreparedness', 'DrainageSystems',
                    'CoastalVulnerability', 'Landslides', 'Watersheds',
                    'DeterioratingInfrastructure', 'PopulationScore', 'WetlandLoss',
                    'InadequatePlanning', 'PoliticalFactors', 'FloodProbability']

    # Keep intersection with available columns
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    if not numeric_cols:
        raise ValueError("None of the expected numeric columns were found in the CSV. Please check column names.")

    # Infer and interpolate
    df = df.infer_objects(copy=False)
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
    df.dropna(subset=numeric_cols, inplace=True)

    # Min-max scale the numeric columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Feature engineering (safe checks for missing columns)
    if 'Urbanization' in df.columns and 'Rainfall' in df.columns:
        df['Urban_Rain_Interaction'] = df['Urbanization'] * df['Rainfall']
    if 'ClimateChange' in df.columns and 'Deforestation' in df.columns:
        df['Climate_Deforestation'] = df['ClimateChange'] * df['Deforestation']
    topographic_components = [c for c in ['Slope', 'DrainageSystems', 'Landslides'] if c in df.columns]
    if topographic_components:
        df['Topographic_Risk'] = df[topographic_components].sum(axis=1)

    # Composite risk index — use available columns and fall back to zeros for missing
    comp_cols = ['Urbanization', 'Rainfall', 'Deforestation', 'DrainageSystems', 'ClimateChange', 'WetlandLoss', 'FloodProbability']
    comp_present = [c for c in comp_cols if c in df.columns]
    if comp_present:
        weights = {
            'Urbanization': 0.2, 'Rainfall': 0.2, 'Deforestation': 0.1, 'DrainageSystems': 0.1,
            'ClimateChange': 0.1, 'WetlandLoss': 0.1, 'FloodProbability': 0.2
        }
        df['Composite_Risk_Index'] = sum(df[c].fillna(0) * weights.get(c, 0) for c in comp_present)

    # Determine target
    if 'FloodLabel' in df.columns:
        df['FloodLabel'] = df['FloodLabel'].astype(int)
    elif 'FloodProbability' in df.columns:
        df['FloodLabel'] = (df['FloodProbability'] > 0.5).astype(int)
    elif 'Flood' in df.columns:
        df['FloodLabel'] = df['Flood'].astype(int)
    else:
        raise ValueError("No target column found. Please include 'FloodLabel' or 'FloodProbability' or 'Flood' in the CSV.")

    # Build features list from numeric + engineered columns (exclude FloodProbability)
    exclude = {'FloodProbability', 'FloodLabel', 'Flood', 'index'}
    feature_cols = [c for c in df.columns if c not in exclude and (df[c].dtype.kind in 'fiu' or c.endswith('_Interaction') or c.endswith('_Index') or c.startswith('PCA_'))]
    if not feature_cols:
        raise ValueError("No feature columns found after preprocessing.")

    X = df[feature_cols]
    y = df['FloodLabel']

    print(f"Using {len(feature_cols)} features. Sample: {feature_cols[:8]}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train RandomForest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    r2 = r2_score(y_test, y_prob)
    rmse = np.sqrt(mean_squared_error(y_test, y_prob))
    mae = mean_absolute_error(y_test, y_prob)

    print(f"Accuracy: {acc*100:.2f}%")
    print(f"R2 (on probabilities): {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cm)

    # Save model and predictions
    model_path = os.path.join(output_dir, 'rf_model.pkl')
    joblib.dump(model, model_path)
    print('Saved model to', model_path)

    # Save predictions with index aligned to original DataFrame
    preds_df = pd.DataFrame({
        'index': X_test.index,
        'probability': y_prob,
        'prediction': y_pred
    })
    preds_csv = os.path.join(output_dir, 'predictions.csv')
    preds_df.to_csv(preds_csv, index=False)
    print('Saved predictions to', preds_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run flood prediction training and output results.')
    parser.add_argument('--csv', type=str, default='backend/datasets/floodpredictiondataset.csv', help='Path to CSV dataset')
    parser.add_argument('--out', type=str, default='backend/output', help='Output directory')
    args = parser.parse_args()

    main(args.csv, args.out)

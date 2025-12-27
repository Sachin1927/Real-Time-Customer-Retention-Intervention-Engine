import pandas as pd
import joblib
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def load_model(env='production'):
    """Loads model from either 'staging' or 'production' folder."""
    base_path = Path(f"models/{env}")
    # Find any joblib file in that folder
    try:
        model_path = list(base_path.glob("*.joblib"))[0]
        logger.info(f"Loading {env} model from: {model_path}")
        return joblib.load(model_path)
    except IndexError:
        logger.error(f"No model found in models/{env}!")
        sys.exit(1)

def make_predictions(input_file, output_file, env='production'):
    """Runs batch predictions."""
    logger.info("Loading data...")
    # Load data (Parquet or CSV)
    if input_file.endswith('.parquet'):
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file)
    
    # Select features (Ensure these match training!)
    features = [
        "age", "tenure_months", "monthly_charges", 
        "avg_daily_usage_min", "payment_fails_last_3m", 
        "last_interaction_sentiment"
    ]
    
    # Load Model
    model = load_model(env)
    
    # Predict
    logger.info(f"Scoring {len(df)} customers...")
    probs = model.predict_proba(df[features])[:, 1]
    
    # Save Results
    results = df[['customer_id']].copy()
    results['churn_probability'] = probs
    results['risk_label'] = results['churn_probability'].apply(lambda x: 'High Risk' if x > 0.5 else 'Safe')
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"✅ Predictions saved to {output_path}")

if __name__ == "__main__":
    # Usage: python src/models/predict_model.py [production/staging]
    env = sys.argv[1] if len(sys.argv) > 1 else 'production'
    
    make_predictions(
        input_file="data/raw/telco_churn_simulated.parquet",
        output_file="data/processed/batch_predictions.csv",
        env=env
    )
import pandas as pd
import xgboost as xgb
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score
import joblib
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_training_data():
    """Fetches historical features from the Feature Store."""
    logger.info("Connecting to Feature Store...")
    fs = FeatureStore(repo_path="data/feature_store")
    
    # Load raw to get IDs
    entity_df = pd.read_parquet('data/raw/telco_churn_simulated.parquet')
    entity_df = entity_df[['customer_id', 'event_timestamp']]
    
    logger.info("Retrieving historical features...")
    training_df = fs.get_historical_features(
        entity_df=entity_df,
        features=[
            "churn_features:age",
            "churn_features:tenure_months",
            "churn_features:monthly_charges",
            "churn_features:avg_daily_usage_min",
            "churn_features:payment_fails_last_3m",
            "churn_features:last_interaction_sentiment",
            "churn_features:churn"
        ]
    ).to_df()
    
    return training_df.dropna()

def train():
    # 1. Prepare Data
    df = get_training_data()
    X = df.drop(columns=['churn', 'event_timestamp', 'customer_id'])
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Train Model
    logger.info("Training XGBoost Model...")
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    
    # 3. Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, probs)
    precision = precision_score(y_test, preds)
    logger.info(f"✅ Model Trained. AUC: {auc:.4f} | Precision: {precision:.4f}")
    
    # 4. Save Model (The Critical Part)
    output_path = Path("models/production")
    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path / "xgb_churn_v1.joblib")
    logger.info(f"💾 Model saved to {output_path}")

if __name__ == "__main__":
    train()
import pandas as pd
import numpy as np
from pathlib import Path

def generate_customer_profiles(n=5000):
    np.random.seed(42)
    
    # 1. Generate Random Base Features
    df = pd.DataFrame({
        'customer_id': [f'CUST_{i}' for i in range(n)],
        'age': np.random.randint(18, 90, n),
        'tenure_months': np.random.randint(0, 72, n),
        'monthly_charges': np.random.uniform(20, 120, n),
        'avg_daily_usage_min': np.random.uniform(0, 500, n),
        'payment_fails_last_3m': np.random.choice([0, 1, 2, 3], size=n, p=[0.8, 0.1, 0.05, 0.05]),
        'last_interaction_sentiment': np.random.uniform(-1, 1, n),
        # --- FIX: Create BOTH required timestamp columns ---
        'event_timestamp': pd.Timestamp.now(),
        'created_timestamp': pd.Timestamp.now()
    })
    
    # 2. Apply "Real World" Logic (The Signal)
    def calculate_risk(row):
        score = 0.1 # Base churn risk (10%)
        
        # Risk Multipliers
        if row['payment_fails_last_3m'] >= 1: score += 0.50
        if row['last_interaction_sentiment'] < -0.5: score += 0.30
        if row['tenure_months'] < 6: score += 0.20
        if row['monthly_charges'] > 100: score += 0.10
        
        # Loyalty Discounts
        if row['tenure_months'] > 24: score -= 0.20
        if row['avg_daily_usage_min'] > 200: score -= 0.10
        
        return np.clip(score, 0, 1)

    df['churn_probability'] = df.apply(calculate_risk, axis=1)
    df['churn'] = df.apply(lambda row: 1 if np.random.rand() < row['churn_probability'] else 0, axis=1)
    
    return df.drop(columns=['churn_probability'])

if __name__ == "__main__":
    output_path = Path("data/raw")
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = generate_customer_profiles(5000)
    df.to_parquet(output_path / "telco_churn_simulated.parquet")
    print("✅ Generated 5,000 profiles with valid timestamps!")
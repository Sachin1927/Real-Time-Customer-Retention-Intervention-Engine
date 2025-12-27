import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path

def plot_performance(model_path, data_path, output_folder="docs/images"):
    """Generates evaluation plots for the README."""
    
    # Load resources
    model = joblib.load(model_path)
    df = pd.read_parquet(data_path)
    
    features = ["age", "tenure_months", "monthly_charges", "avg_daily_usage_min", "payment_fails_last_3m", "last_interaction_sentiment"]
    X = df[features]
    y = df["churn"]
    
    # Predict
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    # Setup Output
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix (Production Model)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{output_folder}/confusion_matrix.png")
    print(f"Saved: {output_folder}/confusion_matrix.png")
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f"{output_folder}/roc_curve.png")
    print(f"Saved: {output_folder}/roc_curve.png")

if __name__ == "__main__":
    plot_performance(
        model_path="models/production/xgb_churn_v1.joblib",
        data_path="data/raw/telco_churn_simulated.parquet"
    )
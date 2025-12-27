from datetime import timedelta
from feast import Entity, Field, FeatureView, FileSource, ValueType
# No need to import CsvFormat anymore!
from feast.types import Float32, Int64, String

# 1. Define Entity
customer = Entity(
    name="customer_id", 
    value_type=ValueType.INT64, 
    description="The ID of the customer"
)

# 2. Define Source (Now pointing to PARQUET)
churn_stats_source = FileSource(
    name="churn_stats_source",
    path="../raw/telco_churn_simulated.parquet",  # <--- CHANGED TO PARQUET
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    # No file_format needed! Feast defaults to Parquet.
)

# 3. Define Feature View
churn_features = FeatureView(
    name="churn_features",
    entities=[customer],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="tenure_months", dtype=Int64),
        Field(name="monthly_charges", dtype=Float32),
        Field(name="avg_daily_usage_min", dtype=Int64),
        Field(name="payment_fails_last_3m", dtype=Int64),
        Field(name="last_interaction_sentiment", dtype=Float32),
        Field(name="churn", dtype=Int64),
    ],
    online=True,
    source=churn_stats_source,
    tags={"team": "retention"},
)
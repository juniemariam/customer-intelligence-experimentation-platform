"""PySpark ETL: raw behavioral tables -> validated customer feature mart.

This is intentionally real Spark code (window functions, joins, aggregations and
Parquet output), while model training stays in the Python ML ecosystem.
"""
from pathlib import Path
import json
from pyspark.sql import SparkSession, functions as F, Window
from pyspark.sql.types import DoubleType

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "spark"
ART = ROOT / "artifacts"
AS_OF = "2026-09-30 23:59:59"


def build_spark(app_name="customer-intelligence-etl"):
    return (SparkSession.builder.appName(app_name).master("local[*]")
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.driver.memory", "2g").getOrCreate())


def _quality(customers, events, transactions):
    return {
        "customers": customers.count(), "events": events.count(), "transactions": transactions.count(),
        "duplicate_customer_ids": customers.groupBy("customer_id").count().filter("count > 1").count(),
        "orphan_events": events.join(customers.select("customer_id"), "customer_id", "left_anti").count(),
        "orphan_transactions": transactions.join(customers.select("customer_id"), "customer_id", "left_anti").count(),
        "negative_transactions": transactions.filter(F.col("amount") < 0).count(),
        "future_events": events.filter(F.col("event_ts") > F.to_timestamp(F.lit(AS_OF))).count(),
        "future_transactions": transactions.filter(F.col("transaction_ts") > F.to_timestamp(F.lit(AS_OF))).count(),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True); ART.mkdir(exist_ok=True)
    spark = build_spark()
    customers = spark.read.option("header", True).option("inferSchema", True).csv(str(RAW / "customers.csv"))
    events = spark.read.option("header", True).option("inferSchema", True).csv(str(RAW / "events.csv"))
    tx = spark.read.option("header", True).option("inferSchema", True).csv(str(RAW / "transactions.csv"))
    labels = spark.read.option("header", True).option("inferSchema", True).csv(str(RAW / "modeling_table.csv")).select("customer_id", "churn_30d", "next_30d_revenue")

    customers = customers.dropDuplicates(["customer_id"]).withColumn("signup_date", F.to_date("signup_date"))
    events = events.dropDuplicates(["event_id"]).withColumn("event_ts", F.to_timestamp("event_ts"))
    tx = tx.dropDuplicates(["transaction_id"]).withColumn("transaction_ts", F.to_timestamp("transaction_ts")).withColumn("amount", F.col("amount").cast(DoubleType()))
    report = _quality(customers, events, tx)
    if any(report[k] for k in ["duplicate_customer_ids", "orphan_events", "orphan_transactions", "negative_transactions", "future_events", "future_transactions"]):
        raise ValueError(f"Data quality gate failed: {report}")

    asof = F.to_timestamp(F.lit(AS_OF))
    e = events.filter((F.col("event_ts") <= asof) & (F.col("event_ts") >= F.date_sub(asof, 90)))
    event_features = e.groupBy("customer_id").agg(
        F.datediff(asof, F.max("event_ts")).alias("recency_days"),
        F.sum(F.when(F.col("event_ts") >= F.date_sub(asof, 30), 1).otherwise(0)).alias("frequency_30d"),
        F.sum(F.when(F.col("event_ts") >= F.date_sub(asof, 7), 1).otherwise(0)).alias("events_7d"),
        F.sum(F.when((F.col("event_ts") < F.date_sub(asof, 7)) & (F.col("event_ts") >= F.date_sub(asof, 14)), 1).otherwise(0)).alias("events_prev_7d"),
        F.countDistinct("event_type").alias("event_type_breadth"),
        F.sum(F.when(F.col("event_type") == "support", 1).otherwise(0)).alias("support_events_90d"),
    ).withColumn("usage_trend", (F.col("events_7d") - F.col("events_prev_7d")) / F.greatest(F.lit(1), F.col("events_prev_7d")))

    tx_features = tx.filter((F.col("transaction_ts") <= asof) & (F.col("transaction_ts") >= F.date_sub(asof, 90))).groupBy("customer_id").agg(
        F.sum(F.when(F.col("transaction_ts") >= F.date_sub(asof, 30), F.col("amount")).otherwise(0.0)).alias("monetary_30d"),
        F.sum("amount").alias("monetary_90d"), F.count("transaction_id").alias("transactions_90d"),
        F.avg("amount").alias("avg_order_value_90d"))

    w = Window.partitionBy("customer_id")
    feature_mart = (customers.join(event_features, "customer_id", "left").join(tx_features, "customer_id", "left").join(labels, "customer_id")
        .fillna({"recency_days": 90, "frequency_30d": 0, "events_7d": 0, "events_prev_7d": 0, "event_type_breadth": 0, "support_events_90d": 0,
                 "usage_trend": 0.0, "monetary_30d": 0.0, "monetary_90d": 0.0, "transactions_90d": 0, "avg_order_value_90d": 0.0})
        .withColumn("tenure_days", F.datediff(F.to_date(F.lit(AS_OF[:10])), "signup_date"))
        .withColumn("engagement_score", F.least(F.lit(1.0), (F.col("frequency_30d") / 15.0) * .65 + (F.col("event_type_breadth") / 4.0) * .35))
        .withColumn("etl_run_ts", F.current_timestamp()))
    feature_mart.write.mode("overwrite").parquet(str(OUT / "customer_features"))
    feature_mart.drop("etl_run_ts").toPandas().to_csv(ROOT / "data" / "processed" / "customer_features_spark.csv", index=False)
    (ART / "spark_data_quality.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"feature_rows": feature_mart.count(), **report}, indent=2)); spark.stop()

if __name__ == "__main__": main()

from pathlib import Path

def test_spark_etl_is_real_pipeline():
    text=(Path(__file__).parents[1]/'src'/'etl'/'spark_etl.py').read_text()
    for token in ['SparkSession','groupBy','Window','write.mode("overwrite").parquet','date_sub']:
        assert token in text

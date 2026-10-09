import pandas as pd
from src.data.quality import quality_report

def test_quality_report_detects_duplicate_and_negative_value():
    df=pd.DataFrame({'customer_id':[1,1], 'monetary_30d':[10,-1], 'churn_30d':[0,1], 'next_30d_revenue':[3,4]})
    r=quality_report(df); assert r['duplicate_ids']==1; assert r['negative_monetary']==1

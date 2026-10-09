import pandas as pd

def quality_report(df: pd.DataFrame, id_col='customer_id'):
    numeric=df.select_dtypes(include='number')
    return {
      'rows':len(df),'columns':len(df.columns),'missing_cells':int(df.isna().sum().sum()),
      'duplicate_ids':int(df[id_col].duplicated().sum()) if id_col in df else None,
      'infinite_values':int((~numeric.applymap(lambda x: pd.notna(x) and abs(x)!=float('inf'))).sum().sum()) if not numeric.empty else 0,
      'negative_monetary':int((df.get('monetary_30d',pd.Series(dtype=float))<0).sum()),
      'future_signup_dates':int((pd.to_datetime(df['signup_date'])>pd.Timestamp.now()).sum()) if 'signup_date' in df else None,
      'target_leakage_candidates':[c for c in df.columns if c.lower().startswith(('churn_','next_30d')) and c not in ['churn_30d','next_30d_revenue']]
    }

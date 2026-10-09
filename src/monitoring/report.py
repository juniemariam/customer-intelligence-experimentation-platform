from pathlib import Path
import json,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
def psi(expected,actual,bins=10):
 e=pd.qcut(expected,q=bins,duplicates='drop'); edges=[x.left for x in e.cat.categories]+[e.cat.categories[-1].right]; ec=pd.cut(expected,edges,include_lowest=True).value_counts(normalize=True,sort=False)+1e-6; ac=pd.cut(actual,edges,include_lowest=True).value_counts(normalize=True,sort=False)+1e-6; return float(((ac-ec)*((ac/ec).apply(__import__('numpy').log))).sum())
def main():
 df=pd.read_csv(ROOT/'data'/'raw'/'modeling_table.csv'); ref=df.sample(frac=.5,random_state=1); cur=df.drop(ref.index); cols=['recency_days','frequency_30d','monetary_30d','engagement_score','usage_trend']; report={'missing_rates':cur[cols].isna().mean().to_dict(),'psi':{c:psi(ref[c],cur[c]) for c in cols},'business':{'churn_rate':float(cur.churn_30d.mean()),'revenue_per_user':float(cur.next_30d_revenue.mean())}}; (ROOT/'artifacts'/'monitoring_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()

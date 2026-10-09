from pathlib import Path
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'data'/'raw'; OUT.mkdir(parents=True,exist_ok=True)

def main(n=5000, seed=42):
    rng=np.random.default_rng(seed); ids=np.arange(1,n+1); today=pd.Timestamp('2026-09-30')
    tenure=rng.integers(30,1500,n); signup=today-pd.to_timedelta(tenure,unit='D')
    plan=rng.choice(['free','basic','pro'],n,p=[.28,.47,.25]); region=rng.choice(['west','central','east'],n)
    engagement=np.clip(rng.beta(2.2,2.4,n),.01,.99); base=np.where(plan=='pro',95,np.where(plan=='basic',42,8))
    frequency=np.maximum(1,rng.poisson(2+12*engagement)); recency=np.maximum(0,rng.poisson(3+25*(1-engagement)))
    monetary=np.maximum(0,rng.normal(base*(.5+engagement),base*.25)); trend=rng.normal((engagement-.5)*.25,.18,n)
    logit=-2.0+0.065*recency-2.2*engagement-1.5*trend-0.00045*tenure+(plan=='free')*.65
    p=1/(1+np.exp(-logit)); churn=rng.binomial(1,p)
    next_rev=np.maximum(0, monetary*(.55+.75*engagement)+12*trend+rng.normal(0,10,n)); next_rev*=1-.72*churn
    customers=pd.DataFrame({'customer_id':ids,'signup_date':signup.date,'plan':plan,'region':region,'tenure_days':tenure})
    features=pd.DataFrame({'customer_id':ids,'recency_days':recency,'frequency_30d':frequency,'monetary_30d':monetary.round(2),'engagement_score':engagement.round(4),'usage_trend':trend.round(4),'tenure_days':tenure,'plan':plan,'region':region,'churn_30d':churn,'next_30d_revenue':next_rev.round(2)})
    # Event and transaction tables are intentionally normalized for SQL exercises.
    ev=[]; tx=[]; eid=1; tid=1
    for i in range(n):
        k=min(int(frequency[i]),30)
        for _ in range(k):
            ts=today-pd.Timedelta(days=int(rng.integers(0,30)),hours=int(rng.integers(0,24)))
            ev.append((eid,ids[i],ts,rng.choice(['login','search','feature_use','support'],p=[.45,.2,.3,.05]))); eid+=1
        for _ in range(max(0,int(rng.poisson(max(monetary[i],0)/35)))):
            ts=today-pd.Timedelta(days=int(rng.integers(0,30))); tx.append((tid,ids[i],ts,round(float(max(1,rng.gamma(2,12))),2))); tid+=1
    events=pd.DataFrame(ev,columns=['event_id','customer_id','event_ts','event_type']); transactions=pd.DataFrame(tx,columns=['transaction_id','customer_id','transaction_ts','amount'])
    # randomized retention experiment with heterogeneous effect for new/high-engagement users
    treatment=rng.binomial(1,.5,n); baseline=.22+.38*engagement-.12*churn; uplift=treatment*(.025+.06*(tenure<180)+.035*(engagement>.65)); conv=rng.binomial(1,np.clip(baseline+uplift,.01,.95))
    experiments=pd.DataFrame({'experiment_id':'retention_campaign_v1','customer_id':ids,'variant':np.where(treatment,'treatment','control'),'converted':conv,'segment_hint':np.where(tenure<180,'new',np.where(engagement>.65,'engaged','established'))})
    for name,df in [('customers',customers),('events',events),('transactions',transactions),('modeling_table',features),('experiments',experiments)]: df.to_csv(OUT/f'{name}.csv',index=False)
    print(f'Generated {n:,} customers, {len(events):,} events, {len(transactions):,} transactions.')
if __name__=='__main__': main()

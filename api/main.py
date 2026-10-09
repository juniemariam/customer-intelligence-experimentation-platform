from pathlib import Path
import json, joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts'; app=FastAPI(title='Customer Intelligence API',version='1.0.0')
class CustomerFeatures(BaseModel):
 recency_days:float; frequency_30d:float; monetary_30d:float; engagement_score:float; usage_trend:float; tenure_days:float; plan:str; region:str
def load(name):
 p=ART/name
 if not p.exists(): raise HTTPException(503,'Models not trained. Run: make data && make train')
 return joblib.load(p)
@app.get('/health')
def health(): return {'status':'ok'}
@app.post('/predict/churn')
def churn(x:CustomerFeatures):
 m=load('churn_model.joblib'); p=float(m.predict_proba(pd.DataFrame([x.model_dump()]))[:,1][0]); return {'churn_probability':p,'high_risk':p>=.5}
@app.post('/predict/value')
def value(x:CustomerFeatures): return {'predicted_next_30d_revenue':max(0,float(load('value_model.joblib').predict(pd.DataFrame([x.model_dump()]))[0]))}
@app.get('/segments/{customer_id}')
def segment(customer_id:int):
 p=ROOT/'data'/'processed'/'segments.csv'
 if not p.exists(): raise HTTPException(503,'Segments not generated')
 x=pd.read_csv(p); r=x[x.customer_id==customer_id]
 if r.empty: raise HTTPException(404,'Customer not found')
 return {'customer_id':customer_id,'segment':int(r.iloc[0].segment)}
@app.get('/experiments/{experiment_id}')
def experiment(experiment_id:str):
 if experiment_id!='retention_campaign_v1': raise HTTPException(404,'Experiment not found')
 p=ART/'experiment_results.json'
 if not p.exists(): raise HTTPException(503,'Run experiment analysis first')
 return json.loads(p.read_text())

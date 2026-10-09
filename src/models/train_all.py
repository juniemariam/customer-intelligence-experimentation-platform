from pathlib import Path
import json, warnings, joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,precision_score,recall_score,brier_score_loss,mean_absolute_error,mean_squared_error,r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
try:
 from xgboost import XGBClassifier,XGBRegressor
except Exception: XGBClassifier=XGBRegressor=None
warnings.filterwarnings('ignore')
try:
 from src.models.mlflow_utils import log_metrics_run
except Exception:
 log_metrics_run=None
ROOT=Path(__file__).resolve().parents[2]; DATA=ROOT/'data'/'raw'/'modeling_table.csv'; ART=ROOT/'artifacts'; ART.mkdir(exist_ok=True)
FEATURES=['recency_days','frequency_30d','monetary_30d','engagement_score','usage_trend','tenure_days','plan','region']; NUM=FEATURES[:6]; CAT=FEATURES[6:]
def prep(): return ColumnTransformer([('num',StandardScaler(),NUM),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),CAT)])
def main():
 df=pd.read_csv(DATA); X=df[FEATURES]; yc=df.churn_30d; yr=df.next_30d_revenue
 Xtr,Xte,ytr,yte=train_test_split(X,yc,test_size=.25,stratify=yc,random_state=42)
 clfs={'logistic':LogisticRegression(max_iter=1500),'random_forest':RandomForestClassifier(n_estimators=250,min_samples_leaf=4,class_weight='balanced',random_state=42)}
 if XGBClassifier: clfs['xgboost']=XGBClassifier(n_estimators=250,max_depth=4,learning_rate=.05,subsample=.85,colsample_bytree=.85,eval_metric='logloss',random_state=42)
 cm={}; best=None; best_auc=-1
 for name,m in clfs.items():
  p=Pipeline([('prep',prep()),('model',m)]).fit(Xtr,ytr); prob=p.predict_proba(Xte)[:,1]; pred=(prob>=.5).astype(int)
  met={'roc_auc':roc_auc_score(yte,prob),'pr_auc':average_precision_score(yte,prob),'f1':f1_score(yte,pred),'precision':precision_score(yte,pred,zero_division=0),'recall':recall_score(yte,pred),'brier':brier_score_loss(yte,prob)}; cm[name]={k:round(float(v),4) for k,v in met.items()}
  if log_metrics_run:
   try: log_metrics_run(f'churn-{name}', 'classification', met, {'model':name})
   except Exception: pass
  if met['roc_auc']>best_auc: best_auc=met['roc_auc']; best=(name,p)
 joblib.dump(best[1],ART/'churn_model.joblib')
 Xtr,Xte,ytr,yte=train_test_split(X,yr,test_size=.25,random_state=42)
 regs={'elastic_net':ElasticNet(alpha=.05,l1_ratio=.3,random_state=42),'random_forest':RandomForestRegressor(n_estimators=250,min_samples_leaf=3,random_state=42)}
 if XGBRegressor: regs['xgboost']=XGBRegressor(n_estimators=250,max_depth=4,learning_rate=.05,subsample=.85,colsample_bytree=.85,random_state=42)
 rm={}; best=None; best_rmse=1e99
 for name,m in regs.items():
  p=Pipeline([('prep',prep()),('model',m)]).fit(Xtr,ytr); pred=p.predict(Xte); met={'mae':mean_absolute_error(yte,pred),'rmse':mean_squared_error(yte,pred)**.5,'r2':r2_score(yte,pred)}; rm[name]={k:round(float(v),4) for k,v in met.items()}
  if log_metrics_run:
   try: log_metrics_run(f'value-{name}', 'regression', met, {'model':name})
   except Exception: pass
  if met['rmse']<best_rmse: best_rmse=met['rmse']; best=(name,p)
 joblib.dump(best[1],ART/'value_model.joblib')
 segcols=['recency_days','frequency_30d','monetary_30d','engagement_score','tenure_days','usage_trend']; Z=StandardScaler().fit_transform(df[segcols]); km=KMeans(n_clusters=4,n_init=20,random_state=42).fit(Z); df['segment']=km.labels_; pca=PCA(2,random_state=42).fit_transform(Z); df['pca1']=pca[:,0]; df['pca2']=pca[:,1]; df[['customer_id','segment','pca1','pca2']].to_csv(ROOT/'data'/'processed'/'segments.csv',index=False); joblib.dump((segcols,km),ART/'segment_model.joblib')
 summary=df.groupby('segment')[segcols].mean().round(2).to_dict('index')
 out={'classification':cm,'best_classifier':best[0] if False else max(cm,key=lambda k:cm[k]['roc_auc']),'regression':rm,'segments':summary}; (ART/'metrics.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()

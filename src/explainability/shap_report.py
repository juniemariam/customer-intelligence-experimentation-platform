from pathlib import Path
import json, joblib, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import shap
ROOT=Path(__file__).resolve().parents[2]; ART=ROOT/'artifacts'; FIG=ROOT/'reports'/'figures'; FIG.mkdir(parents=True,exist_ok=True)
FEATURES=['recency_days','frequency_30d','monetary_30d','engagement_score','usage_trend','tenure_days','plan','region']
def explain(model_file, task, sample_n=400):
    model=joblib.load(ART/model_file); df=pd.read_csv(ROOT/'data'/'raw'/'modeling_table.csv'); X=df[FEATURES].sample(min(sample_n,len(df)),random_state=42)
    prep=model.named_steps['prep']; est=model.named_steps['model']; Z=prep.transform(X); names=prep.get_feature_names_out()
    explainer=shap.Explainer(est, Z); sv=explainer(Z)
    values=sv.values
    if values.ndim==3: values=values[:,:,1]
    imp=pd.DataFrame({'feature':names,'mean_abs_shap':np.abs(values).mean(axis=0)}).sort_values('mean_abs_shap',ascending=False)
    imp.to_csv(ART/f'{task}_shap_importance.csv',index=False)
    shap.summary_plot(values, Z, feature_names=names, show=False, max_display=12); plt.tight_layout(); plt.savefig(FIG/f'{task}_shap_summary.png',dpi=160,bbox_inches='tight'); plt.close()
    return imp.head(10).to_dict('records')
def main():
    out={}
    for f,t in [('churn_model.joblib','churn'),('value_model.joblib','value')]:
        try: out[t]=explain(f,t)
        except Exception as e: out[t]={'status':'skipped','reason':str(e)}
    (ART/'shap_report.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()

from pathlib import Path
import mlflow
ROOT=Path(__file__).resolve().parents[2]
def configure_mlflow():
    tracking=(ROOT/'mlruns').resolve().as_uri(); mlflow.set_tracking_uri(tracking); mlflow.set_experiment('customer-intelligence-platform'); return tracking
def log_metrics_run(run_name, model_family, metrics, params=None, artifacts=None):
    configure_mlflow()
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag('model_family',model_family); mlflow.log_params(params or {}); mlflow.log_metrics({k:float(v) for k,v in metrics.items()})
        for a in artifacts or []:
            if Path(a).exists(): mlflow.log_artifact(str(a))

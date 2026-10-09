PYTHON=python
setup:
	$(PYTHON) -m pip install -r requirements.txt
data:
	$(PYTHON) -m src.data.generate_data
etl:
	$(PYTHON) -m src.etl.spark_etl
train:
	$(PYTHON) -m src.models.train_all
experiment:
	$(PYTHON) -m src.experiments.ab_test
explain:
	$(PYTHON) -m src.explainability.shap_report
monitor:
	$(PYTHON) -m src.monitoring.report
test:
	pytest -q
run: data train experiment monitor
full: data etl train experiment explain monitor test
api:
	uvicorn api.main:app --reload
mlflow:
	mlflow ui --backend-store-uri ./mlruns --port 5000

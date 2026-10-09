import json
from pathlib import Path
import streamlit as st
ROOT=Path(__file__).resolve().parents[1]; st.title('Customer Intelligence & Experimentation')
for title,file in [('Model metrics','metrics.json'),('Experiment','experiment_results.json'),('Monitoring','monitoring_report.json')]:
 st.header(title); p=ROOT/'artifacts'/file; st.json(json.loads(p.read_text())) if p.exists() else st.info('Run pipeline to generate this artifact.')

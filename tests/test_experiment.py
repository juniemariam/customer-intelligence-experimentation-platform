import pandas as pd
from src.experiments.ab_test import analyze
def test_ab_analysis_positive_lift():
 df=pd.DataFrame({'variant':['control']*100+['treatment']*100,'converted':[1]*20+[0]*80+[1]*35+[0]*65})
 r=analyze(df); assert r['absolute_lift']>0; assert 0<=r['p_value']<=1

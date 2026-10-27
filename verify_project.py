"""Integration checks: temporal isolation, saved predictions, unknown categories, UI."""
import json
import numpy as np
import pandas as pd
import joblib
from src.data import ROOT, FEATURES, make_features, load_data, split_data

def main():
    _,clean,audit=load_data()
    train,val,test=split_data(clean)
    assert train.release_date.max()<val.release_date.min()<test.release_date.min()
    assert set(train.id).isdisjoint(val.id) and set(val.id).isdisjoint(test.id)
    assert set(train.id).isdisjoint(test.id)
    assert not any(x in FEATURES for x in ['revenue','success','vote_average','vote_count','popularity'])
    assert clean.budget.ge(1000).all() and clean.revenue.gt(0).all()
    assert (clean.success == clean.revenue.ge(2*clean.budget).astype(int)).all()
    reg=joblib.load(ROOT/'models/revenue_pipeline.joblib')
    clf=joblib.load(ROOT/'models/success_pipeline.joblib')
    saved=pd.read_csv(ROOT/'reports/test_predictions.csv')
    assert saved.id.tolist()==test.id.tolist()
    np.testing.assert_allclose(np.maximum(reg.predict(make_features(test)),0),saved.predicted_revenue,rtol=1e-10)
    np.testing.assert_allclose(clf.predict_proba(make_features(test))[:,1],saved.success_probability,rtol=1e-10)
    scenario=test.iloc[:1].copy()
    scenario['director']='Unseen director'; scenario['primary_company']='New company'
    scenario['original_language']='xx'; scenario['runtime']=np.nan
    X=make_features(scenario)
    assert np.isfinite(reg.predict(X)).all()
    p=clf.predict_proba(X)[0,1]
    assert 0<=p<=1
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=60)
    assert not app.exception, str(app.exception)
    for page in ['Predict a movie','Explore the data','Model evaluation','Methodology']:
        app.sidebar.radio[0].set_value(page).run(timeout=60)
        assert not app.exception, f'{page}: {app.exception}'
    app.sidebar.radio[0].set_value('Predict a movie').run(timeout=60)
    app.button[0].click().run(timeout=60)
    assert not app.exception, str(app.exception)
    assert len(app.metric)==3
    result={'status':'passed','checks':['chronological partitions and disjoint IDs',
            'forbidden feature exclusion','financial cleaning and target rule',
            'saved-model predictions reproduce exported test results',
            'unknown categories and missing runtime prediction','all five app pages',
            'interactive prediction form'],'modeling_rows':audit['modeling_records']}
    (ROOT/'reports/verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()

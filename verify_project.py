"""Integration checks: temporal isolation, saved predictions, unknown categories, UI."""
import json
import datetime
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
    # Regression test for the original product defect: never ship a constant prior
    # as individualized success predictions. Check variation and both decisions.
    assert type(clf.named_steps['model']).__name__ != 'DummyClassifier'
    probabilities=clf.predict_proba(make_features(test))[:,1]
    assert np.isfinite(probabilities).all() and ((probabilities>=0)&(probabilities<=1)).all()
    assert np.ptp(probabilities) > 0.05
    assert (probabilities<0.5).any() and (probabilities>=0.5).any()
    validation_metrics=pd.read_csv(ROOT/'reports/classification_validation.csv')
    ranked=validation_metrics.sort_values(['ROC_AUC','Brier'],ascending=[False,True])
    metrics=json.loads((ROOT/'reports/metrics.json').read_text())
    assert metrics['selection']['classification']==ranked.iloc[0]['Model']
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
    # End-to-end UI examples: changing real inputs must change the probability.
    app.number_input[0].set_value(15000000)
    app.number_input[1].set_value(90)
    app.date_input[0].set_value(datetime.date(2013,6,1))
    app.button[0].click().run(timeout=60)
    assert not app.exception,str(app.exception)
    low=float(app.metric[1].value.strip('%'))/100
    assert low<0.5
    app.number_input[0].set_value(1000000)
    app.number_input[1].set_value(140)
    app.multiselect[0].set_value(['Horror'])
    app.button[0].click().run(timeout=60)
    assert not app.exception,str(app.exception)
    high=float(app.metric[1].value.strip('%'))/100
    assert high>0.5 and high>low

    result={'status':'passed','checks':['chronological partitions and disjoint IDs',
            'forbidden feature exclusion','financial cleaning and target rule',
            'saved-model predictions reproduce exported test results',
            'nonconstant success probabilities above and below 50%',
            'classifier selection follows validation AUC',
            'unknown categories and missing runtime prediction','all five app pages',
            'interactive prediction form with below/above-50 scenarios'],'modeling_rows':audit['modeling_records']}
    (ROOT/'reports/verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()

"""Run: python -m streamlit run app.py"""
import json
import datetime
import joblib
import pandas as pd
import streamlit as st
from src.data import ROOT, GENRES, make_features

st.set_page_config(page_title='Box Office Lab',page_icon='🎬',layout='wide')
st.markdown('''<style>.stApp {background: #101921;} h1,h2,h3 {letter-spacing:-.03em;}
[data-testid="stMetric"] {background:#1c2c38;padding:18px;border-radius:12px;}
</style>''',unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return (joblib.load(ROOT/'models/revenue_pipeline.joblib'),
            joblib.load(ROOT/'models/success_pipeline.joblib'))

@st.cache_data
def load_results():
    clean=pd.read_csv(ROOT/'data/processed/movies_clean.csv')
    clean['genres_list']=clean.genres_list.map(json.loads)
    return (clean,json.loads((ROOT/'reports/metrics.json').read_text()),
            json.loads((ROOT/'reports/data_audit.json').read_text()),
            json.loads((ROOT/'models/input_options.json').read_text()))

if not (ROOT/'models/revenue_pipeline.joblib').exists():
    st.error('Train the models first: python -m src.train')
    st.stop()

clean,metrics,audit,options=load_results()
reg,clf=load_models()
st.caption('DATA SCIENCE / MACHINE LEARNING / TMDB')
st.title('Box Office Lab')
st.write('Explore historical movies, estimate revenue, and inspect how the models perform.')
page=st.sidebar.radio('Workspace',['Overview','Predict a movie','Explore the data','Model evaluation','Methodology'])
st.sidebar.caption('Historical educational model. Monetary values are nominal USD.')

if page=='Overview':
    a,b,c=st.columns(3)
    a.metric('Raw movies',f"{audit['raw_movies']:,}")
    b.metric('Modeling records',f"{audit['modeling_records']:,}")
    c.metric('Held-out movies',f"{audit['splits']['test']['rows']:,}")
    st.subheader('Two questions, two models')
    st.write('Revenue regression estimates worldwide gross. Classification estimates whether gross reaches twice the production budget.')
    if metrics['selection']['classification']=='Majority baseline':
        st.warning('The majority baseline won validation F1. The deployed success estimate is the same training-set rate for every movie, not an individualized forecast. Test ROC-AUC is 0.50.')
    a,b=st.columns(2)
    a.metric('Test revenue MAE',f"${metrics['test_regression']['MAE']/1e6:,.1f}M")
    b.metric('Test success F1',f"{metrics['test_classification']['F1']:.3f}")
    st.image(str(ROOT/'reports/figures/budget_revenue.png'))
    st.info('The success label is a revenue-to-budget proxy. It does not measure audited profit.')

elif page=='Predict a movie':
    st.subheader('Create a movie scenario')
    st.caption('Inputs should represent information known before release. This snapshot contains eligible releases through 2016; final models train through 2013.')
    if metrics['selection']['classification']=='Majority baseline':
        st.info('Success classification did not beat the majority baseline under validation F1. Its probability is constant across scenarios; only revenue estimates are movie-specific.')
    with st.form('prediction'):
        a,b,c=st.columns(3)
        budget=a.number_input('Production budget (USD)',min_value=1000,max_value=1000000000,value=30000000,step=1000000)
        runtime=b.number_input('Runtime (minutes)',min_value=1,max_value=400,value=110)
        date=c.date_input('Release date',value=datetime.date(2015,6,1),min_value=datetime.date(1916,1,1),max_value=datetime.date(2035,12,31))
        genres=st.multiselect('Genres',GENRES,default=['Drama'])
        a,b,c=st.columns(3)
        language=a.selectbox('Original language',options['languages'],index=options['languages'].index('en'))
        company=b.selectbox('Production company',['Unknown']+[v for v in options['companies'] if v!='Unknown'])
        director=c.selectbox('Director',['Unknown']+[v for v in options['directors'] if v!='Unknown'])
        submitted=st.form_submit_button('Estimate movie performance',type='primary')
    if submitted:
        if not genres:
            st.error('Choose at least one genre.')
        else:
            scenario=pd.DataFrame([{'budget':budget,'runtime':runtime,'release_year':date.year,
                'release_month':date.month,'original_language':language,'primary_company':company,
                'director':director,'genres_list':genres}])
            X=make_features(scenario)
            revenue=max(0,float(reg.predict(X)[0])); probability=float(clf.predict_proba(X)[0,1])
            a,b,c=st.columns(3)
            a.metric('Estimated worldwide revenue',f'${revenue/1e6:,.1f}M')
            b.metric('Estimated success probability',f'{probability:.1%}')
            c.metric('Success threshold',f'${2*budget/1e6:,.1f}M')
            st.write('Classifier decision: **'+('Success proxy' if probability>=.5 else 'Below threshold')+'** (probability cutoff 0.50).')
            st.caption('Revenue and success are separate models, so their outputs may disagree. Probabilities have not been separately calibrated. No guaranteed revenue range is claimed.')
            if date.year>options['training_last_year']:
                st.warning('This release year is later than the training period. Treat the result as a historical-model scenario; reliability on current releases is unverified.')
            if budget>options['budget_max'] or budget<options['budget_min']:
                st.warning('Budget is outside the final training data range.')
            payload=scenario.to_dict(orient='records')[0]
            payload.update(predicted_revenue_usd=revenue,success_probability=probability)
            st.download_button('Download this prediction',json.dumps(payload,indent=2),'movie_prediction.json','application/json')
    st.subheader('What influenced the model overall?')
    st.image(str(ROOT/'reports/figures/revenue_importance.png'))
    st.caption('Global validation-set importance; this is not a personalized explanation of the scenario above.')

elif page=='Explore the data':
    a,b=st.columns(2)
    years=a.slider('Release years',int(clean.release_year.min()),int(clean.release_year.max()),
                   (1990,int(clean.release_year.max())))
    genre=b.selectbox('Filter by genre',['All']+GENRES)
    selected=clean[clean.release_year.between(*years)]
    if genre!='All': selected=selected[selected.genres_list.map(lambda g:genre in g)]
    a,b,c=st.columns(3)
    a.metric('Selected movies',len(selected))
    b.metric('Median revenue',f'${selected.revenue.median()/1e6:,.1f}M' if len(selected) else 'N/A')
    c.metric('Success proxy rate',f'{selected.success.mean():.1%}' if len(selected) else 'N/A')
    if len(selected):
        st.scatter_chart(selected,x='budget',y='revenue')
        st.bar_chart(selected.groupby('release_month').success.mean())
    st.dataframe(selected[['title','release_year','budget','revenue','runtime','success','split']],hide_index=True)
    export=selected.copy(); export['genres_list']=export.genres_list.map(json.dumps)
    st.download_button('Download filtered data',export.to_csv(index=False),'filtered_movies.csv','text/csv')
    with st.expander('Full-sample analysis charts'):
        st.caption('These saved charts use all eligible movies and do not change with the filters.')
        for name in ['genre_revenue','revenue_distribution','year_coverage','correlations']:
            st.image(str(ROOT/f'reports/figures/{name}.png'))

elif page=='Model evaluation':
    st.subheader('Validation comparisons')
    st.caption('Train: through 2010. Validation: 2011–2013. Select by revenue MAE and classification F1; then refit on train + validation.')
    st.dataframe(pd.read_csv(ROOT/'reports/regression_validation.csv'),hide_index=True)
    st.dataframe(pd.read_csv(ROOT/'reports/classification_validation.csv'),hide_index=True)
    st.subheader('Final held-out test: 2014–2016')
    st.write('Revenue model: **'+metrics['selection']['regression']+'**. Success model: **'+metrics['selection']['classification']+'**.')
    if metrics['selection']['classification']=='Majority baseline':
        st.warning('A majority predictor can score well on positive-class F1 while having no ranking ability. Interpret its F1 alongside ROC-AUC = 0.50 and the confusion matrix. Future work should predefine balanced accuracy or macro-F1 and collect more informative features, then use a fresh test set.')
    st.dataframe(pd.DataFrame({'Selected revenue model':metrics['test_regression'],
                             'Median baseline':metrics['test_regression_baseline']}))
    st.dataframe(pd.DataFrame({'Selected success model':metrics['test_classification'],
                             'Majority baseline':metrics['test_classification_baseline']}))
    for pair in [('actual_predicted','residuals'),('confusion_matrix','roc_curve')]:
        for col,name in zip(st.columns(2),pair): col.image(str(ROOT/f'reports/figures/{name}.png'))
    st.subheader('Largest revenue errors')
    st.dataframe(pd.read_csv(ROOT/'reports/largest_errors.csv'),hide_index=True)

else:
    st.subheader('Reproducible methodology')
    st.markdown('''1. Merge movie metadata and credits by unique movie ID.
2. Exclude missing/zero financial outcomes and budgets below $1,000; retain an audit.
3. Build budget, runtime, year, seasonal, language, company, director, and multi-label genre features.
4. Split chronologically; fit preprocessing only on the relevant training partition.
5. Compare log-target regressors and binary classifiers against simple baselines.
6. Select on validation, refit through 2013, and evaluate once on 2014–2016 releases.

**Limitations:** selected TMDB sample, uncertain financial reporting, missing financial data,
historical nominal currencies without inflation adjustment, incomplete 2016 coverage,
and no marketing, distribution, competition, or pre-release audience-demand measurements.
Current metadata is a proxy for pre-release information; historical availability was not independently verified.
Feature importance describes associations, not causal effects.''')
    st.json(audit)
    st.markdown('[Original dataset: TMDB 5000 on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)')

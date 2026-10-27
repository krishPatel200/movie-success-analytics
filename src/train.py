"""Run with python -m src.train. Validation selects; final test only evaluates."""
import json
import platform
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    brier_score_loss, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.inspection import permutation_importance
from src.data import ROOT, NUMERIC, CATEGORICAL, GENRE_COLS, make_features, load_data, split_data

SEED = 42
REPORTS = ROOT / 'reports'
FIGURES = REPORTS / 'figures'

def preprocessing():
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), NUMERIC),
        ('category', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=8,
                                  sparse_output=False), CATEGORICAL),
        ('genres', 'passthrough', GENRE_COLS)])

def regression(model, log=True):
    estimator = TransformedTargetRegressor(regressor=model, func=np.log1p,
                    inverse_func=np.expm1) if log else model
    return Pipeline([('prepare', preprocessing()), ('model', estimator)])

def classification(model):
    return Pipeline([('prepare', preprocessing()), ('model', model)])

def reg_metrics(y, prediction):
    prediction = np.maximum(prediction, 0)
    return {'MAE': float(mean_absolute_error(y, prediction)),
            'RMSE': float(np.sqrt(mean_squared_error(y, prediction))),
            'R2': float(r2_score(y, prediction))}

def cls_metrics(y, prediction, probability):
    return {'Accuracy': float(accuracy_score(y, prediction)),
            'Precision': float(precision_score(y, prediction, zero_division=0)),
            'Recall': float(recall_score(y, prediction, zero_division=0)),
            'F1': float(f1_score(y, prediction, zero_division=0)),
            'ROC_AUC': float(roc_auc_score(y, probability)),
            'Brier': float(brier_score_loss(y, probability))}

def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGURES / (name + '.png'), dpi=150, bbox_inches='tight')
    plt.close()

def eda(raw, clean):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'figure.figsize': (9, 5), 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold'})
    plt.hist(np.log10(clean.revenue), bins=35, color='#267c91')
    plt.xlabel('log10 revenue (nominal USD)'); plt.ylabel('Movies')
    plt.title('Revenue spans several orders of magnitude'); savefig('revenue_distribution')
    plt.scatter(clean.budget, clean.revenue, alpha=.35, s=12, color='#267c91')
    plt.xscale('log'); plt.yscale('log'); plt.xlabel('Budget (USD, log scale)')
    plt.ylabel('Revenue (USD, log scale)'); plt.title('Budget and revenue'); savefig('budget_revenue')
    genres = clean[['genres_list','revenue','success']].explode('genres_list')
    stats = genres.groupby('genres_list').agg(count=('revenue','size'),
                   median_revenue=('revenue','median'), success_rate=('success','mean'))
    stats.to_csv(REPORTS / 'genre_summary.csv')
    stats.loc[stats['count'] >= 30].sort_values('median_revenue').median_revenue.div(1e6).plot.barh(color='#267c91')
    plt.xlabel('Median revenue (USD millions)'); plt.ylabel('Genre')
    plt.title('Genre comparison (at least 30 movies; movies may have multiple genres)'); savefig('genre_revenue')
    clean.groupby('release_month').success.mean().reindex(range(1,13)).plot.bar(color='#d48b35')
    plt.xlabel('Release month'); plt.ylabel('Success proxy rate'); plt.ylim(0,1)
    plt.title('Historical success rate by release month'); savefig('monthly_success')
    clean.groupby('release_year').size().plot(color='#267c91')
    plt.xlabel('Release year'); plt.ylabel('Eligible movies in this sample')
    plt.title('Dataset coverage over time'); savefig('year_coverage')
    plt.scatter(clean.runtime, np.log10(clean.revenue), s=12, alpha=.3, color='#267c91')
    plt.xlabel('Runtime (minutes)'); plt.ylabel('log10 revenue (USD)')
    plt.title('Runtime and revenue'); savefig('runtime_revenue')
    raw[['budget','revenue','runtime','release_date','overview']].isna().sum().plot.bar(color='#d48b35')
    plt.ylabel('Missing values (zero financial values counted separately)')
    plt.title('Blank-value audit'); savefig('missing_values')
    corr = clean[['budget','revenue','runtime','release_year','success']].corr(method='spearman')
    fig, ax = plt.subplots(); im=ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)),corr.columns,rotation=35,ha='right')
    ax.set_yticks(range(len(corr)),corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)): ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center')
    fig.colorbar(im,ax=ax); ax.set_title('Spearman associations, not causal effects'); savefig('correlations')

def main():
    for folder in [REPORTS, FIGURES, ROOT/'models',ROOT/'data/processed']: folder.mkdir(parents=True, exist_ok=True)
    raw, clean, audit = load_data()
    train, validation, test = split_data(clean)
    parts = {'train':train, 'validation':validation, 'test':test}
    audit['splits'] = {name:{'rows':len(df),'from':str(df.release_date.min().date()),
                           'to':str(df.release_date.max().date()),'success_rate':float(df.success.mean())}
                       for name,df in parts.items()}
    audit['source_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (ROOT/'data/raw').glob('*.csv')}
    (REPORTS/'data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    compact = clean[['id','title','budget','revenue','runtime','release_date','release_year',
                     'release_month','original_language','genres_list','primary_company','director','success']].copy()
    compact['split'] = np.select([compact.release_year.le(2010), compact.release_year.le(2013)],['train','validation'],default='test')
    compact['genres_list'] = compact.genres_list.map(json.dumps)
    compact.to_csv(ROOT/'data/processed/movies_clean.csv',index=False)
    eda(raw,clean)
    Xtr, Xv, Xt = [make_features(df) for df in [train,validation,test]]
    regressors = {
        'Median baseline':regression(DummyRegressor(strategy='median'),False),
        'Ridge':regression(Ridge(alpha=10)),
        'Random Forest':regression(RandomForestRegressor(n_estimators=180,max_depth=16,min_samples_leaf=3,n_jobs=-1,random_state=SEED)),
        'Gradient Boosting':regression(GradientBoostingRegressor(n_estimators=180,learning_rate=.04,max_depth=2,min_samples_leaf=10,loss='huber',random_state=SEED))}
    classifiers = {
        'Majority baseline':classification(DummyClassifier(strategy='prior')),
        'Logistic Regression':classification(LogisticRegression(C=.3,max_iter=2500,random_state=SEED)),
        'Random Forest':classification(RandomForestClassifier(n_estimators=180,max_depth=12,min_samples_leaf=5,n_jobs=-1,random_state=SEED))}
    reg_rows=[]; cls_rows=[]
    for name, model in regressors.items():
        print('Validation regression:',name,flush=True)
        model.fit(Xtr,train.revenue)
        reg_rows.append({'Model':name,**reg_metrics(validation.revenue,model.predict(Xv))})
    for name, model in classifiers.items():
        print('Validation classification:',name,flush=True)
        model.fit(Xtr,train.success)
        cls_rows.append({'Model':name,**cls_metrics(validation.success,model.predict(Xv),model.predict_proba(Xv)[:,1])})
    reg_table=pd.DataFrame(reg_rows).sort_values('MAE'); cls_table=pd.DataFrame(cls_rows).sort_values('F1',ascending=False)
    reg_table.to_csv(REPORTS/'regression_validation.csv',index=False)
    cls_table.to_csv(REPORTS/'classification_validation.csv',index=False)
    # Keep baseline eligible: a complex model must earn its selection.
    reg_name=reg_table.iloc[0].Model; cls_name=cls_table.iloc[0].Model
    combined=pd.concat([train,validation],ignore_index=True); Xdev=make_features(combined)
    reg=clone(regressors[reg_name]).fit(Xdev,combined.revenue)
    clf=clone(classifiers[cls_name]).fit(Xdev,combined.success)
    predictions=np.maximum(reg.predict(Xt),0); probability=clf.predict_proba(Xt)[:,1]
    labels=clf.predict(Xt)
    baseline_reg=DummyRegressor(strategy='median').fit(np.zeros((len(combined),1)),combined.revenue)
    baseline_cls=DummyClassifier(strategy='prior').fit(np.zeros((len(combined),1)),combined.success)
    result={'selection':{'regression':reg_name,'classification':cls_name},
            'test_regression':reg_metrics(test.revenue,predictions),
            'test_regression_baseline':reg_metrics(test.revenue,baseline_reg.predict(np.zeros((len(test),1)))),
            'test_classification':cls_metrics(test.success,labels,probability),
            'test_classification_baseline':cls_metrics(test.success,baseline_cls.predict(np.zeros((len(test),1))),baseline_cls.predict_proba(np.zeros((len(test),1)))[:,1]),
            'classification_threshold':0.5,'seed':SEED,
            'versions':{'python':platform.python_version(),'sklearn':sklearn.__version__,'pandas':pd.__version__,'numpy':np.__version__}}
    (REPORTS/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    joblib.dump(reg,ROOT/'models/revenue_pipeline.joblib')
    joblib.dump(clf,ROOT/'models/success_pipeline.joblib')
    errors=test[['id','title','release_date','budget','revenue','success']].copy()
    errors['predicted_revenue']=predictions; errors['absolute_error']=abs(test.revenue-predictions)
    errors['success_probability']=probability; errors['predicted_success']=labels
    errors.to_csv(REPORTS/'test_predictions.csv',index=False)
    errors.sort_values('absolute_error',ascending=False).head(20).to_csv(REPORTS/'largest_errors.csv',index=False)
    fig,ax=plt.subplots(); ax.scatter(test.revenue/1e6,predictions/1e6,alpha=.5,s=18,color='#267c91')
    limit=max(test.revenue.max(),predictions.max())/1e6; ax.plot([0,limit],[0,limit],'--',color='#d48b35')
    ax.set(xlabel='Actual revenue (USD millions)',ylabel='Predicted revenue (USD millions)',title='Held-out test: actual versus predicted'); savefig('actual_predicted')
    plt.scatter(predictions/1e6,(test.revenue-predictions)/1e6,alpha=.5,s=18,color='#267c91')
    plt.axhline(0,color='#d48b35',ls='--'); plt.xlabel('Predicted revenue (USD millions)')
    plt.ylabel('Actual minus predicted (USD millions)'); plt.title('Held-out test residuals'); savefig('residuals')
    ConfusionMatrixDisplay.from_predictions(test.success,labels,display_labels=['Below threshold','Success proxy'],cmap='Blues')
    plt.title('Held-out test confusion matrix'); savefig('confusion_matrix')
    RocCurveDisplay.from_predictions(test.success,probability,name=cls_name)
    plt.plot([0,1],[0,1],'--',color='gray'); plt.title('Held-out test ROC curve'); savefig('roc_curve')
    # Interpretation uses validation data and train-only fitted models, not test-driven selection.
    for kind,model,score,target in [('revenue',regressors[reg_name],'neg_mean_absolute_error',validation.revenue),
                                    ('success',classifiers[cls_name],'f1',validation.success)]:
        importance=permutation_importance(model,Xv,target,scoring=score,n_repeats=3,random_state=SEED,n_jobs=1)
        table=pd.DataFrame({'Feature':Xv.columns,'Importance':importance.importances_mean,'Std':importance.importances_std}).sort_values('Importance',ascending=False)
        table.to_csv(REPORTS/f'{kind}_importance.csv',index=False)
        top=table.head(10).sort_values('Importance'); plt.barh(top.Feature,top.Importance,color='#267c91')
        plt.xlabel('Increase in MAE (USD)' if kind=='revenue' else 'Decrease in F1')
        plt.title(f'{kind.title()}: validation permutation importance'); savefig(f'{kind}_importance')
    options={'languages':sorted(clean.original_language.unique().tolist()),
             'companies':sorted(clean.primary_company.value_counts().head(80).index.tolist()),
             'directors':sorted(clean.director.value_counts().head(100).index.tolist()),
             'budget_min':float(combined.budget.min()),'budget_max':float(combined.budget.max()),
             'training_last_year':int(combined.release_year.max())}
    (ROOT/'models/input_options.json').write_text(json.dumps(options,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()

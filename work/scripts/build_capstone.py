from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/"data/raw/content_refresh_anonymized.csv"
OUT=ROOT/"work/outputs"
OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(RAW)
df["label"]=df["trend_direction"].astype(str).str.lower().eq("down").astype(int)
df["prior_ctr"]=np.where(df["impressions_prev_30d"]>0,df["clicks_prev_30d"]/df["impressions_prev_30d"]*100,0)
df["prior_session_rate"]=np.where(df["impressions_prev_30d"]>0,df["sessions_prev_30d"]/df["impressions_prev_30d"]*100,0)
df["content_age_at_start"]=np.maximum(pd.to_numeric(df["content_age_days"],errors="coerce").fillna(0)-30,0)
num=["competition","cpc","word_count","char_count","impressions_prev_30d","clicks_prev_30d","sessions_prev_30d","prior_ctr","prior_session_rate","content_age_at_start"]
cat=["competition_level","content_type","main_intent","word_count_tier"]
for c in num: df[c]=pd.to_numeric(df[c],errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(0)
for c in cat: df[c]=df[c].fillna("unknown").astype(str)
Xn=df[num].copy()
Xn["log_search_volume"]=np.log1p(pd.to_numeric(df["search_volume"],errors="coerce").fillna(0))
for c in ["impressions_prev_30d","clicks_prev_30d","sessions_prev_30d"]: Xn["log_"+c]=np.log1p(Xn[c])
Xn=Xn.drop(columns=["impressions_prev_30d","clicks_prev_30d","sessions_prev_30d"])
X=pd.concat([Xn,pd.get_dummies(df[cat],prefix=cat,dtype=float)],axis=1)
y=df["label"].astype(int)
clients=df["client_id"].fillna("unknown").astype(str)
rng=np.random.default_rng(42); ids=clients.drop_duplicates().to_numpy()
test_clients=set(rng.permutation(ids)[:max(1,int(round(len(ids)*.2)))])
mask=clients.isin(test_clients).to_numpy(); tr=np.where(~mask)[0]; te=np.where(mask)[0]
def p_at_k(y,s,k):
    idx=np.argsort(-np.asarray(s))[:k]; return float(np.asarray(y)[idx].mean())
def pct(s): return pd.Series(s).rank(method="average",pct=True).to_numpy()
baseline=.55*pct(np.log1p(df["impressions_prev_30d"]))+.25*pct(np.log1p(df["sessions_prev_30d"]))+.15*(1-pct(df["word_count"].fillna(0)))+.05*pct(df["competition"].fillna(0))
models={
"logistic_regression":Pipeline([("scaler",StandardScaler()),("model",LogisticRegression(class_weight="balanced",max_iter=1000,random_state=42))]),
"decision_tree":DecisionTreeClassifier(class_weight="balanced",max_depth=5,min_samples_leaf=50,random_state=42),
"random_forest":RandomForestClassifier(class_weight="balanced_subsample",max_depth=10,min_samples_leaf=25,n_estimators=200,n_jobs=-1,random_state=42)}
results={}
for name,m in models.items():
    m.fit(X.iloc[tr],y.iloc[tr]); p=m.predict_proba(X.iloc[te])[:,1]; yt=y.iloc[te]
    results[name]={"accuracy":float(accuracy_score(yt,p>=.5)),"precision":float(precision_score(yt,p>=.5,zero_division=0)),"recall":float(recall_score(yt,p>=.5,zero_division=0)),"f1":float(f1_score(yt,p>=.5,zero_division=0)),"roc_auc":float(roc_auc_score(yt,p)),"average_precision":float(average_precision_score(yt,p)),"precision_at_20":p_at_k(yt,p,20),"precision_at_50":p_at_k(yt,p,50),"precision_at_100":p_at_k(yt,p,100)}
results["baseline_rules"]={"roc_auc":float(roc_auc_score(y.iloc[te],baseline[te])),"average_precision":float(average_precision_score(y.iloc[te],baseline[te])),"precision_at_20":p_at_k(y.iloc[te],baseline[te],20),"precision_at_50":p_at_k(y.iloc[te],baseline[te],50),"precision_at_100":p_at_k(y.iloc[te],baseline[te],100)}
rf=RandomForestClassifier(class_weight="balanced_subsample",max_depth=10,min_samples_leaf=25,n_estimators=200,n_jobs=-1,random_state=42).fit(X,y)
summary={"dataset_rows":int(len(df)),"positive_rate":float(y.mean()),"train_rows":int(len(tr)),"test_rows":int(len(te)),"random_state":42,"feature_count":int(X.shape[1]),"best_model":"random_forest","models":results,"top_features":pd.DataFrame({"feature":X.columns,"importance":rf.feature_importances_}).sort_values("importance",ascending=False).head(15).to_dict("records")}
(OUT/"capstone_metrics.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary["models"],indent=2))

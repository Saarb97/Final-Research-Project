#%%
# Taken from https://www.kaggle.com/code/suvroo/complete-nlp-pipeline#RoPE-(Robust-Positional-Embeddings)

import re
import nltk
nltk.download('stopwords')

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('IMDB Dataset.csv')
#%%
print(df.duplicated().sum())
#%%
df.drop_duplicates(inplace=True)
#%%

def remove_tags(raw_text):
    cleaned_text = re.sub(re.compile('<.*?>'), '', raw_text)
    return cleaned_text
df['review'] = df['review'].apply(remove_tags)

sw_list = stopwords.words('english')
df['review'] = df['review'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(lambda x:" ".join(x))

df['review'] = df['review'].apply(lambda x:x.lower())
#%%
# df = df.sample(1000)
#%%
X = df.iloc[:,0:1]
y = df['sentiment']
#%%
encoder = LabelEncoder()
y = encoder.fit_transform(y)
#%%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
print(X_train.shape)
#%%
# Applying BoW
cv = CountVectorizer()
X_train_bow = cv.fit_transform(X_train['review']).toarray()
X_test_bow = cv.transform(X_test['review']).toarray()

X_train_bow.shape
#%%
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

rf.fit(X_train_bow,y_train)
y_pred = rf.predict(X_test_bow)
accuracy_score(y_test,y_pred)
#%%
X_train_bow
#%%
import optuna

import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000, step=50),
        "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),  # Learning rate
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),  # L2 regularization
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),  # L1 regularization
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),  # Minimum loss reduction
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "verbosity": 0,
        "tree_method": "hist",  # Enable GPU
        "device": "cuda",  # Specify CUDA device
    }

    # Train-validation split
    X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(
        X_train_bow, y_train, test_size=0.2, random_state=42
    )

    dtrain = xgb.DMatrix(X_train_split, label=y_train_split)
    dvalid = xgb.DMatrix(X_valid_split, label=y_valid_split)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        evals=[(dvalid, "eval")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # Predict on validation set
    preds = model.predict(dvalid, iteration_range=(0, model.best_iteration))

    # Compute roc_auc score
    return roc_auc_score(y_valid_split, preds)


# Run Optuna optimization
pruner = optuna.pruners.HyperbandPruner()
study = optuna.create_study(direction="maximize",pruner=pruner)
study.optimize(objective,timeout=3600)

# Best hyperparameters found
best_params = study.best_params
print(f"\n🔹 Best Hyperparameters:\n{best_params}")

# Train final model with best hyperparameters on GPU
best_params["tree_method"] = "gpu_hist"
best_params["device"] = "cuda"

dtrain_cv = xgb.DMatrix(X_train_bow, label=y_train)
final_model = xgb.train(
    params=best_params,
    dtrain=dtrain_cv,
)

# Evaluate on Hold-Out Validation Set
dvalid = xgb.DMatrix(X_test_bow, label=y_test)
test_preds = final_model.predict(dvalid)
accuracy_score(y_test,test_preds)
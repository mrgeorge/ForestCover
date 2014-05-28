import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

if __name__ == "__main__":

    dataDir = "/Users/mgeorge/kaggle/forest/data/"

    # TRAIN
    loc_train = dataDir + "train.csv"
    df_train = pd.read_csv(loc_train)

    feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]

    X_train = df_train[feature_cols]
    y = df_train['Cover_Type']

    rf = RandomForestClassifier(n_estimators = 500, n_jobs = -1)

    rf.fit(X_train, y)

    # CROSS-VALIDATE
    parameters = {"n_estimators":np.logspace(0,3,num=4).astype("int64"),
                  "criterion":("gini", "entropy"),
                  "max_features":np.linspace(0.1,1.,num=4),
                  "bootstrap":(True, False),
                  "oob_score":(True, False)}
    rfcv = GridSearchCV(rf, parameters, cv=10, refit=True, verbose=100, n_jobs=8)

    rfcv.fit(X_train, y)

    # TEST
    loc_test = dataDir + "test.csv"
    df_test = pd.read_csv(loc_test)

    X_test = df_test[feature_cols]
    test_ids = df_test['Id']

    loc_submission = dataDir + "submission.csv"
    with open(loc_submission, "wb") as outfile:
        outfile.write("Id,Cover_Type\n")
        for e, val in enumerate(list(rfcv.predict(X_test))):
            outfile.write("%s,%s\n"%(test_ids[e],val))

import pandas as pd
from sklearn import preprocessing, ensemble
import os
from sklearn import metrics
from . import dispatcher
import joblib
import numpy as np

TESTING_DATA = os.environ.get('TESTING_DATA')
MODEL = os.environ.get("MODEL")


def predict():
    df = pd.read_csv(TESTING_DATA)
    test_idx = df["id"].values
    predictions = None

    for FOLD in range(5):
        df = pd.read_csv(TESTING_DATA)
        encoders = joblib.load(os.path.join("models",f"{MODEL}_{FOLD}_label_encoder.pkl" ))
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))

        for c in encoders:
            lbl = encoders[c]
            df.loc[:,c] = lbl.transform(df[c].values.tolist())


        clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
        df = df[cols]
        preds = clf.predict_proba(df)[:,1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5

    submission = pd.DataFrame(np.column_stack((test_idx, predictions)) , columns=["id", "target"])
    return submission
    

if __name__ == "__main__":
    submission = predict()
    submission.to_csv(f"models/{MODEL}.csv", index= False)

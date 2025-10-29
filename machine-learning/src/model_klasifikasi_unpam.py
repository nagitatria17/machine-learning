"""
model_klasifikasi_unpam.py
Project: Klasifikasi Kelulusan Mahasiswa UNPAM
Author: Maria Nagita Tria Vanessa
NIM: 231011400228
Kelas: 05 TPLE 005

Usage:
    python src/model_klasifikasi_unpam.py --data ../data/kelulusan_unpam.csv --out outputs
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, auc)
import joblib

sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (8,5)

def load_data(path):
    return pd.read_csv(path)

def eda(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir,'info.txt'),'w') as f:
        df.info(buf=f)
    df.describe().to_csv(os.path.join(outdir,'describe.csv'))
    df['Status_Kelulusan'].value_counts().to_csv(os.path.join(outdir,'target_counts.csv'))
    # plots
    plt.figure()
    sns.countplot(x='Status_Kelulusan', data=df)
    plt.title('Distribusi Status Kelulusan'); plt.savefig(os.path.join(outdir,'target_distribution.png')); plt.close()
    plt.figure(figsize=(10,8))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap'); plt.savefig(os.path.join(outdir,'correlation_heatmap.png')); plt.close()

def preprocess(df):
    X = df.drop(columns=['Nama','NIM','Status_Kelulusan'])
    y = (df['Status_Kelulusan']=='Lulus').astype(int)
    numeric_features = ['IPK','Kehadiran','Lama_Studi']
    categorical_features = ['Jenis_Kelamin','Program_Studi']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return X, y, preprocessor

def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, outdir):
    os.makedirs(outdir, exist_ok=True)
    models = {
        'LogisticRegression': LogisticRegression(max_iter=500, random_state=42),
        'DecisionTree': DecisionTreeClassifier(max_depth=6, random_state=42)
    }
    results = {}
    for name, model in models.items():
        pipe = Pipeline(steps=[('pre', preprocessor), ('clf', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe.named_steps['clf'], 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
        else:
            fpr = tpr = roc_auc = None

        # save metrics and artifacts
        results[name] = {'accuracy':acc,'precision':prec,'recall':rec,'f1':f1,'roc_auc':roc_auc,'y_pred':y_pred,'y_proba':y_proba}

        # confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tidak Lulus','Lulus'], yticklabels=['Tidak Lulus','Lulus'])
        plt.title(f'Confusion Matrix - {name}'); plt.savefig(os.path.join(outdir,f'cm_{name}.png')); plt.close()

        # save model
        joblib.dump(pipe, os.path.join(outdir, f'model_{name}.joblib'))

        # export tree text
        if name=='DecisionTree':
            clf = pipe.named_steps['clf']
            # attempt to build feature names
            try:
                cat_feats = pipe.named_steps['pre'].transformers_[1][1].get_feature_names_out()
                feature_names = list(pipe.named_steps['pre'].transformers_[0][2]) + list(cat_feats)
            except Exception:
                feature_names = None
            if feature_names is not None:
                with open(os.path.join(outdir,'decision_tree.txt'),'w') as f:
                    f.write(export_text(clf, feature_names=feature_names))

    # save results summary
    pd.DataFrame({k:{'accuracy':v['accuracy'],'precision':v['precision'],'recall':v['recall'],'f1':v['f1'],'roc_auc':v['roc_auc']} for k,v in results.items()}).to_csv(os.path.join(outdir,'results_summary.csv'))
    return results

def main(args):
    df = load_data(args.data)
    eda(df, args.outdir)
    X, y, preprocessor = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    results = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor, args.outdir)
    print("Results:")
    for k,v in results.items():
        print(k, v['accuracy'], v['f1'], v['roc_auc'])

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/kelulusan_unpam.csv')
    parser.add_argument('--outdir', type=str, default='../outputs')
    args = parser.parse_args()
    main(args)

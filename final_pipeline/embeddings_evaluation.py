#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# imports 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
import os
import glob

# defining functions 
# defining function to load embeddings csv files and output a single dataframe 
def load_embeddings(input_dir: str) -> pd.DataFrame:
    csv_paths = glob.glob(os.path.join(input_dir, "*_embeddings.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"Could not fine file in {input_dir}")
    
    frames = []
    for path in csv_paths:
        subpop_name = os.path.basename(path).replace("_embeddings.csv", "")
        df = pd.read_csv(path)
        df["subpopulation"] = subpop_name
        frames.append(df)
        print(f"  loaded {subpop_name}: {len(df)} clips")
 
    combined = pd.concat(frames, ignore_index=True)
    print(f"Total: {len(combined)} clips across {combined['subpopulation'].nunique()} subpopulations")
    return combined

# defining random forest classifier function 
def rf_classifier(X: np.ndarray, y: np.ndarray, label: str, n_splits: int = 5, random_state: int = 42):
    
    # initilaising classifier and cross-validation
    clf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
 
    # generating predictions using cross-fold validation
    y_pred = cross_val_predict(clf, X, y, cv=cv)
 
    # using predictions from cross validation to test accuracy of random forest classifier 
    acc = accuracy_score(y, y_pred)
    print(f"\n=== {label} ===")
    print(f"Cross-validated accuracy: {acc:.3f}")
    # generating classification report to consider effects of size imbalance 
    print(classification_report(y, y_pred))
 
    # generating confusion matrix 
    labels_sorted = sorted(np.unique(y))
    cm = confusion_matrix(y, y_pred, labels=labels_sorted)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
 
    # creating plot 
    fig, ax = plt.subplots(figsize=(max(6, len(labels_sorted) * 0.8), max(5, len(labels_sorted) * 0.7)))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels_sorted, yticklabels=labels_sorted, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted subpopulation")
    ax.set_ylabel("True subpopulation")
    ax.set_title(f"Confusion matrix (row-normalized) — {label}\naccuracy = {acc:.3f}")
    plt.tight_layout()
    out_path = f"confusion_matrix_{label.replace(' ', '_')}_v2.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved confusion matrix plot to {out_path}")
    plt.close(fig)
 
    return acc, cm

# training classifier on clip loudness as a confound check 
def compute_rmse_descriptor(audio_dir: str, filenames: list, subpop_col: pd.Series) -> np.ndarray:
 
    rmse_vals = []
    for fname, subpop in zip(filenames, subpop_col):
        wav_path = os.path.join(audio_dir, subpop, fname)
        if not os.path.exists(wav_path):
            rmse_vals.append(np.nan)
            continue
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        rmse_vals.append(float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))))
    return np.array(rmse_vals)

def final_evaluate (): 
    
    base_dir = os.path.abspath(os.path.dirname(__file__)) if '__file__' in locals() else os.getcwd()
    embeddings_dir = os.path.join(base_dir, "subpopulation_embeddings_v2")   
    label = "first_pipeline"                
    audio_dir = os.path.join(base_dir, "cleaned_audio_v2")                             
    # number of cv folds
    n_splits = 5                                
    
 
    print(f"Loading embeddings from {embeddings_dir} ...")
    df = load_embeddings(embeddings_dir)
 
    vector_cols = [c for c in df.columns if c.startswith("vector")]
    X = df[vector_cols].values
    y = df["subpopulation"].values
 
    # running main classifier trained on embeddings 
    acc_full, cm_full = rf_classifier(X, y, label=f"{label}_full_embeddings", n_splits= n_splits)
 
    
    if audio_dir:
        # retrieving rmse values
        print("\nComputing RMSE descriptor for confound check")
        rmse = compute_rmse_descriptor(audio_dir, df["Filename"].tolist(), df["subpopulation"])
        valid_mask = ~np.isnan(rmse)
        if valid_mask.sum() < len(rmse):
            print(f"  warning: {(~valid_mask).sum()} clips had no matching .wav file and were skipped")
        X_rmse = rmse[valid_mask].reshape(-1, 1)
        y_rmse = y[valid_mask]
 
        # running rmse classifier as confound check 
        acc_rmse, _ = rf_classifier(X_rmse, y_rmse, label=f"{label}_rmse_only", n_splits= n_splits)
 
        print(f"\n--- Confound comparison ---")
        print(f"Full embedding accuracy : {acc_full:.3f}")
        print(f"RMSE-only accuracy      : {acc_rmse:.3f}")
        if acc_rmse > 0.6 * acc_full:
            print("A large fraction of the classification accuracy can be exlained by RMSE")
        else:
            print("RMSE alone is a weaker predicter than the embeddings. Can skip amplitude normalisation")
    else:
        print("\n(No --audio_dir provided, skipping RMSE confound check.)")
 
    print(f"\Evaluation complete")
 
if __name__ == "__final_evaluate__":
    final_evaluate()

# executing functions 
final_evaluate()


import json
import pickle
from pathlib import Path
 
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
 
from features.extractor import extract_features, get_feature_names


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "ocsvm_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
META_PATH = MODEL_DIR / "model_meta.json"

GAMMA_GRID = ["scale", "auto", 1e-4, 1e-3, 1e-2, 0.1] 
NU_GRID = [0.01, 0.05, 0.1, 0.15]

def build_feature_matrix(domains: list[str]) -> np.ndarray:
    rows = [extract_features(d) for d in domains]
    feature_names = get_feature_names()
    return np.array([[r[k] for k in feature_names] for r in rows], dtype=float)

def save_extracted_matrices(X_legit: np.ndarray, X_dga: np.ndarray, output_dir: str = "data"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    feature_names = get_feature_names()
    
    if len(X_legit) > 0:
        df_legit = pd.DataFrame(X_legit, columns=feature_names)
        legit_file = out_path / "legit_features_extracted.csv"
        df_legit.to_csv(legit_file, index=False)
        print(f"[Info] Saved {len(df_legit)} legit feature records to {legit_file}")
        
    if len(X_dga) > 0:
        df_dga = pd.DataFrame(X_dga, columns=feature_names)
        dga_file = out_path / "dga_features_extracted.csv"
        df_dga.to_csv(dga_file, index=False)
        print(f"[Info] Saved {len(df_dga)} DGA feature records to {dga_file}")


def train(csv_path: str = "data/dns_data.csv", test_size: float = 0.25):
    df = pd.read_csv(csv_path, header=0, names=["label", "family", "domain"]).sample(100000)
    df["domain"] = df["domain"].str.strip().str.lower()
 
    legit_domains = df[df["label"] == "legit"]["domain"].tolist()
    dga_domains = df[df["label"] == "dga"]["domain"].tolist()
 
    ## getting features
    X_legit = build_feature_matrix(legit_domains)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_legit)
    X_train, X_val = train_test_split(X_scaled, test_size=test_size, random_state=42)
 
    ## dga data for prediction
    X_dga = build_feature_matrix(dga_domains) if dga_domains else np.empty((0, X_train.shape[1]))
    save_extracted_matrices(X_legit, X_dga)
    if len(X_dga):
        X_dga_scaled = scaler.transform(X_dga)
        X_val_combined = np.vstack([X_val, X_dga_scaled])
        y_val = np.array([1] * len(X_val) + [-1] * len(X_dga_scaled))
    else:
        X_val_combined = X_val
        y_val = np.ones(len(X_val))
 
    ### hyper parameter 
    best_f1 = -1.0
    best_params = {"gamma": 0.1, "nu": 0.1}
 
    print("[Layer 2] Grid searching (γ, ν) …")
    for gamma in GAMMA_GRID:
        for nu in NU_GRID:
            print(f"loop for gamme: {gamma} and mu: {nu}")
            clf = OneClassSVM(kernel="rbf", gamma=gamma, nu=nu)
            clf.fit(X_train)
            preds = clf.predict(X_val_combined)  # +1 or -1
            if len(np.unique(y_val)) > 1:
                score = f1_score(y_val, preds, pos_label=-1, zero_division=0)
            else:
                score = float(np.mean(preds == y_val))
            if score > best_f1:
                best_f1 = score
                best_params = {"gamma": gamma, "nu": nu}
 
    print(f"[Layer 2] Best params: {best_params}  (val F1={best_f1:.3f})")

    final_clf = OneClassSVM(kernel="rbf", **best_params)
    final_clf.fit(X_scaled)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_clf, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
 
    meta = {
        "best_gamma": best_params["gamma"],
        "best_nu": best_params["nu"],
        "val_f1": round(best_f1, 4),
        "n_train": len(X_scaled),
        "feature_names": get_feature_names(),
    }
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
 
    print(f"[Layer 2] Model saved → {MODEL_PATH}")
    return final_clf, scaler, meta
 
 
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found."
        )
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return clf, scaler
 
 
def predict_raw(domains: list[str], clf, scaler) -> np.ndarray:
    X = build_feature_matrix(domains)
    X_scaled = scaler.transform(X)
    return clf.predict(X_scaled)
 
 
def decision_scores(domains: list[str], clf, scaler) -> np.ndarray:  ## -ve: more anomalous, +ve more normal
    X = build_feature_matrix(domains)
    X_scaled = scaler.transform(X)
    return clf.decision_function(X_scaled)


if __name__ == "__main__":
    # train()
    clf, scaler = load_model()
    test_domains = [
        "www.google.com",          # Legit
        "www.facebook.com",        # Legit
        "www.amazon.com",          # Legit
        "www.mzltrack.com",        # Legit (from dataset)
        "mortiscontrastatim.com",  # DGA (from dataset)
        "randomdomain12345.com",   # Likely DGA
        "xj3k9s8d7f6g5h4j3k2l1.com"  # Likely DGA
    ]
    predictions = predict_raw(test_domains, clf, scaler)
    for d, p in zip(test_domains, predictions): 
        label = "legit" if p == 1 else "dga"
        print(f"Domain: {d:25} | Predicted: {label}")
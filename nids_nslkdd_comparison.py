# =====================================================================
#  NETWORK INTRUSION DETECTION SYSTEM (NIDS) - MACHINE LEARNING COMPARISON
#  Dataset : NSL-KDD  (KDDTrain.parquet  +  KDDTest.parquet)
#  Models  : Random Forest, Decision Tree, XGBoost, Naive Bayes
#  Metrics : Accuracy, Precision, Recall, F1, ROC-AUC, CV score,
#            Training time, Latency, Detection Rate, False Alarm Rate
#
#  HOW TO USE IN GOOGLE COLAB:
#  Every block that starts with "# %%" goes into its own Colab cell.
#  Run the cells from top to bottom.
# =====================================================================


# %% BLOCK 1 - UPLOAD THE DATASET FILES (Colab only)
# ---------------------------------------------------------------------
# WHAT THIS DOES:
#   Opens a file-upload button. Select KDDTrain.parquet and KDDTest.parquet.
#   Colab stores uploaded files in its working folder, so later we only
#   need to write the file NAME (no full path).
# ---------------------------------------------------------------------
from google.colab import files
uploaded = files.upload()          # choose BOTH files in the popup


# %% BLOCK 2 - IMPORT LIBRARIES
# ---------------------------------------------------------------------
# WHAT THIS DOES:
#   Loads all the tools we need:
#   pandas/numpy -> data handling | matplotlib/seaborn -> graphs
#   scikit-learn -> ML models & metrics | xgboost -> XGBoost model
# ---------------------------------------------------------------------
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RANDOM_STATE = 42                  # fixed seed -> same results every run
np.random.seed(RANDOM_STATE)


# %% BLOCK 3 - SETTINGS (the only place you may need to edit)
# ---------------------------------------------------------------------
# WHAT THIS DOES:
#   Keeps all user choices in one place.
#   CLASSIFICATION_MODE = "binary"     -> Normal vs Attack (2 classes)
#   CLASSIFICATION_MODE = "multiclass" -> Normal, DoS, Probe, R2L, U2R (5 classes)
# ---------------------------------------------------------------------
TRAIN_FILE = "KDDTrain.parquet"
TEST_FILE  = "KDDTest.parquet"
CLASSIFICATION_MODE = "binary"     # "binary" or "multiclass"
CV_FOLDS = 5                       # cross-validation folds (0 = skip, faster)


# %% BLOCK 4 - LOAD TRAIN AND TEST DATA
# ---------------------------------------------------------------------
# WHAT THIS DOES:
#   Reads the two parquet files into tables (DataFrames).
#   Because parquet files from different sources can have different
#   column names, this block also:
#     - assigns the standard NSL-KDD column names if the columns are unnamed
#     - finds the label column (the column that says normal / attack type)
#     - removes the 'difficulty' column if present (it is not a real feature)
# ---------------------------------------------------------------------
NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"]

def load_nsl(path):
    df = pd.read_parquet(path)
    # If columns are just numbers (0,1,2...), give them proper names
    if all(isinstance(c, (int, np.integer)) for c in df.columns):
        names = NSL_COLUMNS + ["label", "difficulty"]
        df.columns = names[:df.shape[1]]
    df.columns = [str(c).strip() for c in df.columns]
    # Find the label column
    for cand in ["label", "class", "attack", "target", "Label", "Class"]:
        if cand in df.columns:
            df = df.rename(columns={cand: "label"})
            break
    else:
        df = df.rename(columns={df.columns[-1]: "label"})   # fall back to last column
    # Drop 'difficulty' column if it exists
    df = df.drop(columns=[c for c in df.columns if "difficulty" in c.lower()])
    # Parquet sometimes stores text as bytes -> convert to normal text
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].apply(lambda v: v.decode() if isinstance(v, bytes) else v)
    return df

train_df = load_nsl(TRAIN_FILE)
test_df  = load_nsl(TEST_FILE)
print("Train shape:", train_df.shape, "| Test shape:", test_df.shape)
display(train_df.head())


# %% BLOCK 5 - DATA CLEANING
# ---------------------------------------------------------------------
# WHAT THIS DOES:
#   - Replaces infinite values with empty values and removes rows with
#     missing data (bad records would confuse the models)
#   - Removes DUPLICATE rows from the training set (duplicates make the
#     model memorise repeated records and give biased results)
#   - The official test set is kept as it is, so results are comparable
#     with other research papers
# ---------------------------------------------------------------------
for d in (train_df, test_df):
    d.replace([np.inf, -np.inf], np.nan, inplace=True)

before = len(train_df)
train_df = train_df.dropna().drop_duplicates().reset_index(drop=True)
test_df  = test_df.dropna().reset_index(drop=True)
print(f"Train: removed {before - len(train_df)} duplicate/missing rows")
print("Train:", train_df.shape, "| Test:", test_df.shape)


# %% BLOCK 6 - LABEL PREPARATION (what are we predicting?)
# ---------------------------------------------------------------------
# WHAT THIS DOES:
#   NSL-KDD has ~40 specific attack names (neptune, smurf, nmap ...).
#   We convert them as follows:
#     binary     -> "Normal" or "Attack"
#     multiclass -> Normal, DoS, Probe, R2L, U2R  (the 4 standard attack families)
#   Then the text labels are converted to numbers, because ML models
#   work on numbers.
# ---------------------------------------------------------------------
ATTACK_FAMILY = {}
for a in ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
          "udpstorm", "processtable", "worm", "mailbomb"]:
    ATTACK_FAMILY[a] = "DoS"
for a in ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"]:
    ATTACK_FAMILY[a] = "Probe"
for a in ["guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster",
          "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack",
          "httptunnel", "sendmail", "named"]:
    ATTACK_FAMILY[a] = "R2L"
for a in ["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack",
          "xterm", "ps"]:
    ATTACK_FAMILY[a] = "U2R"

def convert_label(v):
    v = str(v).strip().lower().rstrip(".")
    if v == "normal":
        return "Normal"
    if CLASSIFICATION_MODE == "binary":
        return "Attack"
    return ATTACK_FAMILY.get(v, "Attack")

train_df["target"] = train_df["label"].apply(convert_label)
test_df["target"]  = test_df["label"].apply(convert_label)

label_enc = LabelEncoder().fit(pd.concat([train_df["target"], test_df["target"]]))
class_names = list(label_enc.classes_)
y_train = label_enc.transform(train_df["target"])
y_test  = label_enc.transform(test_df["target"])
n_classes = len(class_names)

print("Classes:", class_names)
print("\nTrain class distribution:\n", train_df["target"].value_counts())
print("\nTest class distribution:\n", test_df["target"].value_counts())

# Graph: class distribution (useful for the 'Dataset' section of the paper)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
train_df["target"].value_counts().plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Training Set - Class Distribution")
test_df["target"].value_counts().plot(kind="bar", ax=axes[1], color="indianred")
axes[1].set_title("Testing Set - Class Distribution")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300)
plt.show()


# %% BLOCK 7 - FEATURE PREPARATION (encoding + scaling)
# ---------------------------------------------------------------------
# WHAT THIS DOES:
#   1. Separates the input features (X) from the label (y).
#   2. Converts the 3 text columns (protocol_type, service, flag) into
#      numbers. The encoder learns from the TRAIN data only; any new value
#      that appears only in the test data is marked as -1.
#   3. Removes columns that never change (useless for learning).
#   4. Scales the values (StandardScaler). This mainly helps Naive Bayes;
#      tree models do not need it, so they use the unscaled data.
# ---------------------------------------------------------------------
X_train = train_df.drop(columns=["label", "target"]).copy()
X_test  = test_df.drop(columns=["label", "target"]).copy()

cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
print("Categorical columns being encoded:", cat_cols)
if cat_cols:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train[cat_cols] = enc.fit_transform(X_train[cat_cols].astype(str))
    X_test[cat_cols]  = enc.transform(X_test[cat_cols].astype(str))

X_train = X_train.astype(float)
X_test  = X_test.astype(float)

constant_cols = [c for c in X_train.columns if X_train[c].nunique() <= 1]
X_train = X_train.drop(columns=constant_cols)
X_test  = X_test.drop(columns=constant_cols)
feature_names = X_train.columns.tolist()
print("Removed constant columns:", constant_cols)
print("Final number of features:", len(feature_names))

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_names)
X_test_s  = pd.DataFrame(scaler.transform(X_test),  columns=feature_names)


# %% BLOCK 8 - DEFINE THE MODELS
# ---------------------------------------------------------------------
# WHAT THIS DOES:
#   Creates the four algorithms we compare.
#   class_weight="balanced" makes the trees pay more attention to rare
#   attack types instead of ignoring them.
# ---------------------------------------------------------------------
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE,
        class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE, class_weight="balanced"),
    "XGBoost": XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        n_jobs=-1, random_state=RANDOM_STATE),
    "Naive Bayes": GaussianNB(),
}


# %% BLOCK 9 - TRAINING, TESTING AND MEASURING PERFORMANCE
# ---------------------------------------------------------------------
# WHAT THIS DOES (for every model):
#   - TRAIN on KDDTrain and measure the training time
#   - TEST on KDDTest (data the model has never seen) and measure latency
#     = time taken to classify one record (important for real-time IDS)
#   - Calculate Accuracy, Precision, Recall, F1-score, ROC-AUC
#   - Run cross-validation on the training data for a more reliable score
#   - Print a detailed per-class report
# ---------------------------------------------------------------------
results, predictions, fitted = [], {}, {}

for name, model in models.items():
    Xtr, Xte = (X_train_s, X_test_s) if name == "Naive Bayes" else (X_train, X_test)

    t0 = time.perf_counter()
    model.fit(Xtr, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(Xte)
    latency_ms = (time.perf_counter() - t0) / len(Xte) * 1000   # ms per record

    try:
        proba = model.predict_proba(Xte)
        auc = (roc_auc_score(y_test, proba[:, 1]) if n_classes == 2
               else roc_auc_score(y_test, proba, multi_class="ovr",
                                  average="weighted"))
    except Exception:
        auc = np.nan

    if CV_FOLDS:
        skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv = cross_val_score(model, Xtr, y_train, cv=skf,
                             scoring="accuracy", n_jobs=-1)
        cv_mean, cv_std = cv.mean(), cv.std()
    else:
        cv_mean = cv_std = np.nan

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "ROC-AUC": auc,
        "CV Acc (mean)": cv_mean,
        "CV Acc (std)": cv_std,
        "Train Time (s)": train_time,
        "Latency (ms/sample)": latency_ms,
    })
    predictions[name], fitted[name] = y_pred, model

    print(f"\n================ {name} ================")
    print(classification_report(y_test, y_pred, labels=range(n_classes),
                                target_names=class_names, zero_division=0))


# %% BLOCK 10 - RESULTS TABLE
# ---------------------------------------------------------------------
# WHAT THIS DOES:
#   Puts all the numbers in one comparison table, saves it as a CSV file
#   (use it as the results table in the paper) and names the best model.
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results).set_index("Model")
display(results_df.round(4))
results_df.to_csv("model_results.csv")
print("Best model by F1-Score:", results_df["F1-Score"].idxmax())


# %% BLOCK 11 - GRAPH 1: ACCURACY COMPARISON
# ---------------------------------------------------------------------
# WHAT THIS DOES: Bar chart comparing test accuracy of the four models.
# ---------------------------------------------------------------------
palette = sns.color_palette("viridis", len(results_df))

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(results_df.index, results_df["Accuracy"] * 100, color=palette)
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
            f"{b.get_height():.2f}%", ha="center", fontweight="bold")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy Comparison")
ax.set_ylim(max(0, results_df["Accuracy"].min() * 100 - 10), 101)
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=300)
plt.show()


# %% BLOCK 12 - GRAPH 2: CLASSIFICATION PERFORMANCE COMPARISON
# ---------------------------------------------------------------------
# WHAT THIS DOES: Grouped bar chart showing Accuracy, Precision, Recall
#   and F1-score of every model side by side.
# ---------------------------------------------------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
ax = (results_df[metrics] * 100).plot(kind="bar", figsize=(10, 6), width=0.8,
                                      colormap="Set2", edgecolor="black")
for c in ax.containers:
    ax.bar_label(c, fmt="%.1f", fontsize=8, padding=2)
ax.set_ylabel("Score (%)")
ax.set_title("Classification Performance Comparison")
ax.set_ylim(max(0, results_df[metrics].min().min() * 100 - 10), 105)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("classification_comparison.png", dpi=300)
plt.show()


# %% BLOCK 13 - GRAPH 3: MODEL LATENCY COMPARISON
# ---------------------------------------------------------------------
# WHAT THIS DOES: Shows how fast each model is.
#   Left  = time to classify ONE record (detection speed)
#   Right = time needed to train the model
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col, title in zip(
        axes, ["Latency (ms/sample)", "Train Time (s)"],
        ["Model Latency Comparison (Inference)", "Training Time Comparison"]):
    bars = ax.bar(results_df.index, results_df[col], color=palette)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{b.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(col)
    ax.set_title(title)
plt.tight_layout()
plt.savefig("latency_comparison.png", dpi=300)
plt.show()


# %% BLOCK 14 - GRAPH 4: RANDOM FOREST - TOP 15 FEATURES + CONFUSION MATRIX
# ---------------------------------------------------------------------
# WHAT THIS DOES:
#   1. Finds the 15 most important features according to Random Forest
#      (which network properties help most in detecting attacks).
#   2. Retrains Random Forest using ONLY those 15 features (a lighter,
#      faster model).
#   3. Draws its confusion matrix: rows = actual class, columns =
#      predicted class. The diagonal shows correct predictions.
# ---------------------------------------------------------------------
importances = pd.Series(fitted["Random Forest"].feature_importances_,
                        index=feature_names)
top15 = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(9, 6))
sns.barplot(x=top15.values, y=top15.index, palette="mako")
plt.title("Top 15 Features (Random Forest Importance)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("top15_feature_importance.png", dpi=300)
plt.show()

rf_top = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                random_state=RANDOM_STATE,
                                class_weight="balanced")
rf_top.fit(X_train[top15.index], y_train)

t0 = time.perf_counter()
y_pred_top = rf_top.predict(X_test[top15.index])
lat_top = (time.perf_counter() - t0) / len(X_test) * 1000

cm = confusion_matrix(y_test, y_pred_top, labels=range(n_classes))
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: Random Forest (Top 15 Features)")
plt.tight_layout()
plt.savefig("confusion_matrix_rf_top15.png", dpi=300)
plt.show()

print(f"RF (Top 15) Accuracy : {accuracy_score(y_test, y_pred_top):.4f}")
print(f"RF (Top 15) F1-Score : {f1_score(y_test, y_pred_top, average='weighted'):.4f}")
print(f"RF (Top 15) Latency  : {lat_top:.5f} ms/sample "
      f"(all features: {results_df.loc['Random Forest','Latency (ms/sample)']:.5f})")


# %% BLOCK 15 - EXTRA GRAPH: CONFUSION MATRICES OF ALL MODELS
# ---------------------------------------------------------------------
# WHAT THIS DOES: Draws a normalized confusion matrix for every model
#   (values are fractions of each actual class), so you can see which
#   model confuses which classes.
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, (name, pred) in zip(axes.ravel(), predictions.items()):
    cm_n = confusion_matrix(y_test, pred, labels=range(n_classes), normalize="true")
    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.suptitle("Normalized Confusion Matrices", fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrices_all.png", dpi=300)
plt.show()


# %% BLOCK 16 - IDS-SPECIFIC METRICS: DETECTION RATE & FALSE ALARM RATE
# ---------------------------------------------------------------------
# WHAT THIS DOES:
#   Treats everything that is not "Normal" as an attack and calculates:
#   Detection Rate (DR)   = attacks correctly caught / all attacks
#   False Alarm Rate (FAR)= normal traffic wrongly flagged / all normal traffic
#   A good IDS has HIGH detection rate and LOW false alarm rate.
# ---------------------------------------------------------------------
normal_idx = class_names.index("Normal")
rows = []
for name, pred in predictions.items():
    actual_attack = (y_test != normal_idx)
    pred_attack   = (pred != normal_idx)
    tn, fp, fn, tp = confusion_matrix(actual_attack, pred_attack).ravel()
    rows.append({"Model": name,
                 "Detection Rate (TPR)": tp / (tp + fn),
                 "False Alarm Rate (FPR)": fp / (fp + tn)})
ids_df = pd.DataFrame(rows).set_index("Model")
display(ids_df.round(4))
ids_df.to_csv("ids_metrics.csv")


# %% BLOCK 17 - EXTRA GRAPH: ROC CURVES (binary mode only)
# ---------------------------------------------------------------------
# WHAT THIS DOES: Plots the ROC curve of each model. The closer a curve
#   is to the top-left corner (and the higher the AUC), the better.
# ---------------------------------------------------------------------
if n_classes == 2:
    plt.figure(figsize=(7, 6))
    for name, model in fitted.items():
        Xte = X_test_s if name == "Naive Bayes" else X_test
        p = model.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, p):.4f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=300)
    plt.show()


# %% BLOCK 18 - DOWNLOAD ALL GRAPHS AND TABLES (Colab only)
# ---------------------------------------------------------------------
# WHAT THIS DOES: Downloads every saved image and CSV to your computer
#   so you can paste them into your research paper.
# ---------------------------------------------------------------------
import glob
for f in glob.glob("*.png") + glob.glob("*.csv"):
    files.download(f)

# %% [markdown]
# # Analiză completă BankChurners (EDA + Model)
# 
# Pași:
# 1) Încărcare & curățare
# 2) EDA (statistici, distribuții, rate churn pe categorii)
# 3) Corelații (numeric)
# 4) Modelare (train/test, pipeline cu OneHot + Scale, LogisticRegression & RandomForest)
# 5) Evaluare (accuracy, precision, recall, f1, AUC, matrice confuzie, importanțe)
# 
# Notă: toate graficele folosesc matplotlib (fără seaborn) și câte un grafic per figură.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#from caas_jupyter_tools import display_dataframe_to_user

# -----------------------
# 1) Încărcare & curățare
# -----------------------
df = pd.read_csv("C:/Users/dariu/Desktop/Proiect 2/BankChurners.csv")

# Elimină coloanele de probabilități Naive Bayes (artefact din sursa originală)
nb_cols = [c for c in df.columns if c.startswith("Naive_Bayes_Classifier")]
df = df.drop(columns=nb_cols, errors="ignore")

# Definim ținta binară: 1=Attrited Customer, 0=Existing Customer
df["churn"] = (df["Attrition_Flag"].str.strip().str.lower() == "attrited customer").astype(int)

# Coloană ID care nu ajută la model
if "CLIENTNUM" in df.columns:
    df = df.drop(columns=["CLIENTNUM"])

# Curățăm spațiile/NA pentru categorice
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
for c in cat_cols:
    df[c] = df[c].fillna("Unknown").astype(str).str.strip()

# Asigurăm numeric pentru cele numerice
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# excluzem ținta churn din lista numerice pentru EDA numerică
num_cols_wo_target = [c for c in num_cols if c != "churn"]

# -----------------------
# 2) EDA de bază
# -----------------------

# a) Dimensiuni & tipuri
shape_info = pd.DataFrame({
    "rows": [df.shape[0]],
    "cols": [df.shape[1]]
})

# b) Lipsuri pe coloană
missing = df.isna().sum().sort_values(ascending=False).to_frame(name="missing_count")
missing["missing_pct"] = (missing["missing_count"] / len(df)).round(4)

# c) Statistici numerice
desc_num = df[num_cols_wo_target].describe().T

# d) Distribuția țintei
churn_counts = df["churn"].value_counts().to_frame(name="count")
churn_counts["percent"] = (churn_counts["count"] / churn_counts["count"].sum() * 100).round(2)
churn_counts.index = churn_counts.index.map({0:"Existing (0)", 1:"Attrited (1)"})

# Afișăm tabele utile în UI
display_dateframe_to_user = print  # Înlocuiți cu funcția reală din mediul dvs.
print(display_dateframe_to_user("Set de date BankChurners - primii 5", df.head()))
print("Dimensiuni & tipuri", shape_info)
print("Valori lipsă pe coloană", missing)
print("Statistici numerice", desc_num)
print("Distribuția churn",churn_counts)

# e) Grafice simple: distribuții pentru câteva variabile cheie
def plot_hist(col, bins=30):
    plt.figure()
    plt.hist(df[col].values, bins=bins)
    plt.title(f"Distribuția {col}")
    plt.xlabel(col)
    plt.ylabel("Frecvență")
    plt.tight_layout()
    plt.show()

for col in ["Customer_Age", "Months_on_book", "Credit_Limit", "Total_Trans_Ct", "Total_Trans_Amt", "Avg_Utilization_Ratio"]:
    if col in df.columns:
        plot_hist(col)

# f) Rata de churn pe categorice cheie
def churn_rate_by(cat_col):
    tbl = (df.groupby(cat_col)["churn"]
           .mean()
           .sort_values(ascending=False)
           .to_frame(name="churn_rate"))
    tbl["count"] = df.groupby(cat_col).size()
    return tbl

cat_to_check = ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]
for c in cat_to_check:
    if c in df.columns:
        display_dateframe_to_user = print 
        display_dataframe_to_user = (f"Rată churn după {c}", df[c].value_counts().to_frame("count"))
        tbl = churn_rate_by(c)
        display_dataframe_to_user = (f"Rată churn după {c}", tbl)

        # Bar chart
        plt.figure()
        plt.bar(tbl.index.astype(str), tbl["churn_rate"].values)
        plt.title(f"Rată churn după {c}")
        plt.ylabel("churn rate")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

# -----------------------
# 3) Corelații (numeric)
# -----------------------
if len(num_cols_wo_target) > 1:
    corr = df[num_cols_wo_target + ["churn"]].corr(numeric_only=True)
    # Heatmap cu imshow
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr.values, aspect="auto")
    plt.xticks(ticks=np.arange(len(corr.columns)), labels=corr.columns, rotation=90)
    plt.yticks(ticks=np.arange(len(corr.index)), labels=corr.index)
    plt.title("Matrice corelații (numeric)")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

# -----------------------
# 4) Modelare
# -----------------------

# Selectăm feature-urile (excludem coloanele text redundante: Attrition_Flag este echivalent cu ținta)
X = df.drop(columns=["churn", "Attrition_Flag"])
y = df["churn"]

# Re-identificăm coloanele categorice / numerice în X
X_cat = X.select_dtypes(include=["object"]).columns.tolist()
X_num = X.select_dtypes(include=[np.number]).columns.tolist()

# Train/Test split stratificat
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Preprocesare
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=False), X_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), X_cat),
    ],
    remainder="drop"
)

# Model 1: Logistic Regression
log_clf = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=200, n_jobs=None))
])

# Model 2: Random Forest
rf_clf = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    ))
])

# Antrenare
log_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)

# 5) Evaluare

def evaluate_model(name, model):
    y_pred = model.predict(X_test)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    # Raport text
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T

    # Matrice confuzie
    cm = confusion_matrix(y_test, y_pred)

    # AUC
    auc = None
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = None

    # Afișăm rezultate
    display_dataframe_to_user(f"[{name}] classification report", report_df)

    # Matrice confuzie plot
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"[{name}] Matrice confuzie")
    plt.xlabel("Predicție")
    plt.ylabel("Adevăr")
    plt.xticks([0,1],[0,1])
    plt.yticks([0,1],[0,1])
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, int(val), ha='center', va='center')
    plt.tight_layout()
    plt.show()

    # ROC
    if y_proba is not None:
        plt.figure()
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"[{name}] Curba ROC")
        plt.tight_layout()
        plt.show()

    return {"report": report_df, "cm": cm, "auc": auc}

res_log = evaluate_model("LogisticRegression", log_clf)
res_rf  = evaluate_model("RandomForest", rf_clf)

# Importanțe / coeficienți
# Pentru LogisticRegression, extragem numele feature-urilor finale
prep = log_clf.named_steps["prep"]
ohe = prep.named_transformers_["cat"]
num_features = X_num
cat_features = list(ohe.get_feature_names_out(X_cat))
feature_names = num_features + cat_features

# Coef logit (un singur vector - problemă binară)
log_coefs = pd.Series(log_clf.named_steps["clf"].coef_.ravel(), index=feature_names).sort_values(ascending=False)
display_dataframe_to_user("Coeficienți LogisticRegression (top 30)", log_coefs.head(30).to_frame("coef"))
display_dataframe_to_user("Coeficienți LogisticRegression (bottom 30)", log_coefs.tail(30).to_frame("coef"))

# Importanțe RandomForest (folosește aceleași feature-uri transformate)
rf_importances = pd.Series(rf_clf.named_steps["clf"].feature_importances_, index=feature_names).sort_values(ascending=False)
display_dataframe_to_user("Importanțe RandomForest (top 30)", rf_importances.head(30).to_frame("importance"))

# Salvăm un mini rezumat în CSV
summary = pd.DataFrame({
    "model": ["LogisticRegression", "RandomForest"],
    "test_AUC": [res_log["auc"], res_rf["auc"]],
    "accuracy": [res_log["report"].loc["accuracy","precision"], res_rf["report"].loc["accuracy","precision"]],
    "recall_churn": [res_log["report"].loc["1","recall"], res_rf["report"].loc["1","recall"]],
    "precision_churn": [res_log["report"].loc["1","precision"], res_rf["report"].loc["1","precision"]],
    "f1_churn": [res_log["report"].loc["1","f1-score"], res_rf["report"].loc["1","f1-score"]],
})
summary_path = "/mnt/data/model_summary.csv"
summary.to_csv(summary_path, index=False)

display_dataframe_to_user("Rezumat performanță modele", summary)

print(f"Fișier rezumat modele salvat la: {summary_path}")

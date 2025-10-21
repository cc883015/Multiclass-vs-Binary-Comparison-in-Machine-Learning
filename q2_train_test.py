import argparse, pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt

from utils_data import load_train_test, detect_label_col, prepare_task_datasets
from metrics_utils import compute_all_metrics

def top1_with_margin(proba: np.ndarray, classes: list, min_prob: float=0.35, margin: float=0.10, fallback=None):
    top_idx = proba.argmax(axis=1)
    top_p = proba.max(axis=1)
      # 找次高概率（partial 排序足够）
    part = np.partition(proba, -2, axis=1)
    second_p = part[:, -2]
    y = []
    for i in range(len(proba)):
        if top_p[i] >= min_prob and (top_p[i] - second_p[i]) >= margin:
            y.append(classes[top_idx[i]])
        else:
            y.append(fallback if fallback is not None else classes[top_idx[i]])
    return np.array(y, dtype=object)

def strategy_class_weights(Xtr, ytr, Xte, yte, classes, pre, model_name):
    if model_name == "LR":
        clf = LogisticRegression(max_iter=200, class_weight="balanced")
    else:
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42, class_weight="balanced")
    pipe = Pipeline([("prep", pre), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    y_pred = pipe.predict(Xte)
    return y_pred, pipe

def strategy_smote(Xtr, ytr, Xte, yte, classes, pre, model_name):
    if model_name == "LR":
        clf = LogisticRegression(max_iter=200)
    else:
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    pipe = ImbPipeline([("prep", pre), ("smote", SMOTE(random_state=42)), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    y_pred = pipe.predict(Xte)
    return y_pred, pipe

def strategy_threshold(Xtr, ytr, Xte, yte, classes, pre, model_name, min_prob=0.35, margin=0.10):
    if model_name == "LR":
        clf = LogisticRegression(max_iter=200)
    else:
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    pipe = Pipeline([("prep", pre), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    Xte_t = pipe.named_steps["prep"].transform(Xte)
    proba = pipe.named_steps["clf"].predict_proba(Xte_t)
    fallback = "Normal" if "Normal" in classes else max(set(classes), key=classes.count)
    y_pred = top1_with_margin(proba, classes=pipe.named_steps["clf"].classes_.tolist(),
                              min_prob=min_prob, margin=margin, fallback=fallback)
    return y_pred, pipe

def plot_bar(df, metric, fname, title):
    import matplotlib.pyplot as plt
    plt.figure()
    labels = [f"{s}-{m}" for s, m in zip(df["strategy"], df["model"])]
    plt.bar(labels, df[metric].values)
    plt.title(title); plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric); plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="训练CSV路径")
    ap.add_argument("--test",  required=True, help="测试CSV路径")
    ap.add_argument("--label", default=None, help="标签列名（可选）")
    args = ap.parse_args()

    df_tr, df_te = load_train_test(args.train, args.test)
    label_col = detect_label_col(df_tr, args.label)

    # 仅多分类
    Xtr, ytr, Xte, yte, classes, pre = prepare_task_datasets(df_tr, df_te, label_col, "multiclass")

    strategies = {
        "class_weights": strategy_class_weights,
        "smote":         strategy_smote,
        "threshold":     strategy_threshold,
    }
    models = ["LR", "RF"]

    rows = []
    for m in models:
        for sname, sfn in strategies.items():
            y_pred, pipe = sfn(Xtr, ytr, Xte, yte, classes, pre, m)
            mtr = compute_all_metrics(yte, y_pred, labels=classes, model=pipe, X_test=Xte)
            mtr.update({"strategy": sname, "model": m})
            rows.append(mtr)
            cm = confusion_matrix(yte, y_pred, labels=classes)
            pd.DataFrame(cm, index=classes, columns=classes).to_csv(f"confusion_{sname}_{m}.csv", index=True)

    out = pd.DataFrame(rows)
    out.to_csv("metrics_q2.csv", index=False)
    print(out)

       # 画图：Macro-F1 与 Macro-FPR
    plot_bar(out, "macro_f1",  "q2_macro_f1.png",  "Q2 Macro-F1 (strategy × model)")
    plot_bar(out, "macro_fpr", "q2_macro_fpr.png", "Q2 Macro-FPR (strategy × model)")

if __name__ == "__main__":
    main()

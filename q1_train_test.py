# q1_train_test.py —— Q1：二分类 vs 多分类（LR/RF），自动选标签 + 强化防泄漏
import argparse, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from utils_data import load_train_test, prepare_task_datasets, pick_labels_for_q1

MODELS = {
    "LR": lambda: LogisticRegression(max_iter=200),
    "RF": lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
}

def run_once(df_tr, df_te, label_col, task):
    Xtr, ytr, Xte, yte, classes, pre = prepare_task_datasets(df_tr, df_te, label_col, task)
    rows = []
    for mname, mfn in MODELS.items():
        pipe = Pipeline([("prep", pre), ("clf", mfn())])
        pipe.fit(Xtr, ytr)
        y_pred = pipe.predict(Xte)

          # 指标
        from metrics_utils import compute_all_metrics
        m = compute_all_metrics(yte, y_pred, labels=classes, model=pipe, X_test=Xte)
        m.update({"setting": task, "model": mname})
        rows.append(m)

          # 混淆矩阵
        cm = confusion_matrix(yte, y_pred, labels=classes)
          pd.DataFrame(cm, index=classes, columns=classes).to_csv(f"cm_{task}_{mname}.csv")
    return pd.DataFrame(rows)

def plot_bar(df, metric, fname, title):
    plt.figure()
    labels = [f"{m}-{s}" for m, s in zip(df["model"], df["setting"])]
    plt.bar(labels, df[metric].values)
    plt.title(title); plt.xticks(rotation=30, ha="right")
    plt.ylabel(metric); plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test",  required=True)
           # 可选：手动指定 binary/multiclass 的标签列
    ap.add_argument("--label_binary", default=None)
    ap.add_argument("--label_multiclass", default=None)
    args = ap.parse_args()

    df_tr, df_te = load_train_test(args.train, args.test)

          # 自动挑选标签
    if args.label_binary is None or args.label_multiclass is None:
        auto_bin, auto_multi = pick_labels_for_q1(df_tr)
        label_bin  = args.label_binary or auto_bin
        label_multi= args.label_multiclass or auto_multi
        print(f"[INFO] binary label = {label_bin}; multiclass label = {label_multi or '(fallback to binary)'}")
    else:
        label_bin, label_multi = args.label_binary, args.label_multiclass

       # 跑二分类
    df_bin = run_once(df_tr, df_te, label_bin, "binary")

       # 跑多分类（如果找不到多分类标签，则退回到与二分类相同的列）
    if label_multi:
        df_multi = run_once(df_tr, df_te, label_multi, "multiclass")
    else:
        df_multi = run_once(df_tr, df_te, label_bin, "multiclass")

    out = pd.concat([df_bin, df_multi], ignore_index=True)
    out.to_csv("metrics_q1.csv", index=False)
    print(out)

        # 画关键指标
    plot_bar(out, "accuracy",              "q1_accuracy.png",  "Q1 Accuracy (LR/RF × Binary/Multiclass)")
    plot_bar(out, "macro_f1",              "q1_macro_f1.png",  "Q1 Macro-F1 (LR/RF × Binary/Multiclass)")
    plot_bar(out, "macro_fpr",             "q1_macro_fpr.png", "Q1 Macro-FPR (LR/RF × Binary/Multiclass)")
    plot_bar(out, "latency_ms_per_sample", "q1_latency.png",   "Q1 Latency ms/sample (LR/RF × Binary/Multiclass)")

if __name__ == "__main__":
    main()

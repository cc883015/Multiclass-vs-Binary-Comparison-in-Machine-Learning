# plot_results.py
# 用法：
#   python3 plot_results.py --q1 metrics_q1.csv --q2 metrics_q2.csv
# 说明：严格使用 matplotlib；一图一指标；不设颜色

import argparse, pandas as pd
import matplotlib.pyplot as plt

def bar(df, xlabels, y, title, out):
    plt.figure()
    plt.bar(xlabels, df[y].values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(y); plt.title(title); plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q1", default=None)
    ap.add_argument("--q2", default=None)
    args = ap.parse_args()

    if args.q1:
        q1 = pd.read_csv(args.q1)
        bar(q1, [f"{m}-{s}" for m,s in zip(q1.model, q1.setting)], "accuracy", "Q1 Accuracy", "q1_accuracy.png")
        bar(q1, [f"{m}-{s}" for m,s in zip(q1.model, q1.setting)], "macro_f1", "Q1 Macro-F1", "q1_macro_f1.png")
        bar(q1, [f"{m}-{s}" for m,s in zip(q1.model, q1.setting)], "macro_fpr","Q1 Macro-FPR","q1_macro_fpr.png")
        bar(q1, [f"{m}-{s}" for m,s in zip(q1.model, q1.setting)], "latency_ms_per_sample","Q1 Latency","q1_latency.png")

    if args.q2:
        q2 = pd.read_csv(args.q2)
        bar(q2, [f"{s}-{m}" for s,m in zip(q2.strategy, q2.model)], "macro_f1","Q2 Macro-F1","q2_macro_f1.png")
        bar(q2, [f"{s}-{m}" for s,m in zip(q2.strategy, q2.model)], "macro_fpr","Q2 Macro-FPR","q2_macro_fpr.png")

if __name__ == "__main__":
    main()

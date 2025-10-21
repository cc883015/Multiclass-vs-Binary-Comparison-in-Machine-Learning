# utils_data.py  —— 强化版数据读取/预处理（自动防泄漏）
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 常见标签/别名
CAND_LABELS = [
    "label","Label",
    "attack_cat","attack",
    "class","category","Category",
    "Attack","Attack_cat"
]
LEAKY_LABELS = set(CAND_LABELS)  # 训练时从 X 中强制移除
DROP_COLS_HINT = {"id","ID","Index","index","Unnamed: 0"}  # 典型 ID/索引列
NORMAL_TOKENS = {
    "normal","Normal","BENIGN","Benign","None","NONE",
    "No Attack","Background","background","BenignTraffic"
}

def detect_label_col(df: pd.DataFrame, user_label: Optional[str]) -> str:
    if user_label is not None:
        assert user_label in df.columns, f"Label column '{user_label}' not found."
        return user_label
    for c in CAND_LABELS:
        if c in df.columns:
            return c
    raise ValueError("无法识别标签列，请用 --label 指定。")

def build_binary_target(y_raw: pd.Series) -> pd.Series:
    """更健壮的二分类目标构造"""
    y_num = pd.to_numeric(y_raw, errors="coerce")
    if y_num.notna().any():
        uniq = set(y_num.dropna().unique().tolist())
        if uniq.issubset({0, 1}):
            return y_num.astype("Int64")
        return (y_num.fillna(0) != 0).astype("Int64")
    normal_set = {s.lower() for s in NORMAL_TOKENS}
    y_str = y_raw.astype(str).str.strip()
    return y_str.apply(lambda v: 0 if v.lower() in normal_set or v == "0" else 1).astype("Int64")

def load_train_test(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(train_path), pd.read_csv(test_path)

def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number","bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", ohe, cat_cols),
        ],
        remainder="drop",
    )

def _drop_obvious_id_and_alias(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = list(DROP_COLS_HINT & set(df.columns))
    return df.drop(columns=to_drop) if to_drop else df

def _drop_leakage_by_rules(X: pd.DataFrame, y: pd.Series) -> List[str]:
    """返回应从 X 中删除的列名（泄漏/常量/高基数）"""
    drops: List[str] = []
    y_str = y.astype(str)

    for c in X.columns:
        col = X[c]
        # 1) 常量列
        if col.nunique(dropna=False) <= 1:
            drops.append(c); continue
        # 2) 高基数且是非数值（基本是 ID 或唯一键）
        if (not pd.api.types.is_numeric_dtype(col)) and (col.nunique() / len(col) > 0.99):
            drops.append(c); continue
        # 3) 与 y 完全一致
        if (col.astype(str) == y_str).all():
            drops.append(c); continue
        # 4) “取值→标签”一对一（每个取值仅对应一个标签）
        gp = pd.crosstab(col.astype(str), y_str)
        if (gp.gt(0).sum(axis=1) <= 1).all():
            drops.append(c); continue
    return drops

def prepare_task_datasets(
    df_train: pd.DataFrame, df_test: pd.DataFrame, label_col: str, task: str
):
    # 0) 先去掉明显 ID/索引
    df_train = _drop_obvious_id_and_alias(df_train)
    df_test  = _drop_obvious_id_and_alias(df_test)

    # 1) 取 y & X，并移除“可疑标签别名列”
    y_train_raw = df_train[label_col]
    y_test_raw  = df_test[label_col]

    leak_alias = list((LEAKY_LABELS - {label_col}) & set(df_train.columns))
    X_train_raw = df_train.drop(columns=[label_col] + leak_alias, errors="ignore")
    X_test_raw  = df_test.drop(columns=[label_col] + leak_alias, errors="ignore")

    # 2) 构造 y
    if task == "binary":
        y_train = build_binary_target(y_train_raw)
        y_test  = build_binary_target(y_test_raw)
        mask_tr = y_train.notna(); mask_te = y_test.notna()
        X_train_raw, y_train = X_train_raw[mask_tr], y_train[mask_tr].astype(int)
        X_test_raw,  y_test  = X_test_raw[mask_te],  y_test[mask_te].astype(int)
        classes = [0, 1]
    else:
        y_train = y_train_raw.astype(str)
        y_test  = y_test_raw.astype(str)
        classes = sorted(pd.concat([y_train, y_test]).dropna().unique().tolist())

    # 3) **训练集**上做自动泄漏检测，并把同样的列从测试集也删掉
    leak_cols = _drop_leakage_by_rules(X_train_raw, y_train)
    if leak_cols:
        X_train_raw = X_train_raw.drop(columns=leak_cols, errors="ignore")
        X_test_raw  = X_test_raw.drop(columns=leak_cols, errors="ignore")
        print(f"[INFO] removed potential leakage columns: {leak_cols}")

    # 4) 仅用训练集构建预处理器（避免任何“窥探”测试集的信息）
    pre = make_preprocessor(X_train_raw)

    return X_train_raw, y_train, X_test_raw, y_test, classes, pre

# ---------- 供 Q1 自动挑选标签的小工具 ----------
def pick_labels_for_q1(df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """返回 (binary_label, multiclass_label)。尽量选不同列，multiclass 有 >2 类。"""
    cand = [c for c in CAND_LABELS if c in df.columns]
    bin_label = None
    multi_label = None

    # 先找显然的二分类标签（0/1 或仅两类）
    for c in cand:
        s = df[c]
        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().any() and set(s_num.dropna().unique()).issubset({0,1}):
            bin_label = c; break
        if s.astype(str).nunique(dropna=True) == 2 and bin_label is None:
            bin_label = c

    # 多分类尽量选 attack_cat 等，且类别数>2，且不同于 bin_label
    for c in cand:
        if c == bin_label: 
            continue
        if df[c].astype(str).nunique(dropna=True) > 2:
            multi_label = c; break

    # 兜底：如果找不到多分类，就 None（由 Q1 脚本退回到和二分类相同）
    return bin_label or cand[0], multi_label

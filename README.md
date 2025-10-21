

# Multiclass vs Binary Comparison in Machine Learning

**Bilingual README (English / 中文双语版)**

---

## **English Version**

### 1. Project Overview

This project explores and compares the **performance differences between binary and multiclass classification models** under identical datasets, features, and computational power.
It also investigates how various **data-balancing and optimization techniques** (such as class weighting, SMOTE oversampling, and threshold tuning) affect the confusion levels in multiclass classification.

Two key research questions:

1. **Q1:** Under the same data and computing resources, how do binary and multiclass classification models differ in performance?

   * Metrics: Accuracy, Macro-F1, Macro-FPR, Latency (ms/sample), Model Size (MB)

2. **Q2:** Can the confusion in multiclass classification be reduced using class weighting, SMOTE, or threshold tuning?

---

### 2. Directory Structure

```
├── q1_train_test.py           # Binary vs Multiclass performance comparison
├── q2_train_test.py           # Confusion reduction using weights, SMOTE, thresholds
├── metrics_utils.py           # Functions for computing metrics
├── utils_data.py              # Dataset loading, preprocessing, and splitting
├── plot_results.py            # Visualization of experiment results
│
├── q1_accuracy.png            # Accuracy comparison
├── q1_latency.png             # Latency comparison
├── q1_macro_f1.png            # Macro-F1 comparison
├── q2_macro_f1.png            # Q2 Macro-F1 comparison (strategies)
├── q2_macro_fpr.png           # Q2 Macro-FPR comparison (strategies)
│
├── cm_binary_LR.csv           # Confusion matrix (Binary - Logistic Regression)
├── cm_binary_RF.csv           # Confusion matrix (Binary - Random Forest)
├── cm_multiclass_LR.csv       # Confusion matrix (Multiclass - Logistic Regression)
├── cm_multiclass_RF.csv       # Confusion matrix (Multiclass - Random Forest)
├── confusion_class_weights_LR.csv  # Confusion matrix after class weighting (LR)
├── confusion_class_weights_RF.csv  # Confusion matrix after class weighting (RF)
│
└── README.md                  # Documentation
```

---

### 3. File Descriptions

| File                 | Description                                                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **q1_train_test.py** | Evaluates Logistic Regression and Random Forest in binary and multiclass settings, saving metrics and confusion matrices. |
| **q2_train_test.py** | Tests three confusion-reduction strategies (class weighting, SMOTE, threshold tuning) and records the results.            |
| **metrics_utils.py** | Defines computation of Accuracy, Precision, Recall, Macro-F1, Macro-FPR, and Latency.                                     |
| **utils_data.py**    | Handles data loading, preprocessing, and dataset splitting.                                                               |
| **plot_results.py**  | Generates plots comparing accuracy, latency, and F1-score between models and strategies.                                  |
| **CSV / PNG files**  | Store metrics, confusion matrices, and visual results for analysis.                                                       |

---

### 4. Metrics Explanation

| Metric                  | Definition                                                                 | Purpose                                          |
| ----------------------- | -------------------------------------------------------------------------- | ------------------------------------------------ |
| **Accuracy**            | Ratio of correct predictions to total predictions.                         | Measures general model correctness.              |
| **Macro-F1**            | Harmonic mean of precision and recall averaged equally across all classes. | Evaluates balance in imbalanced datasets.        |
| **Macro-FPR**           | Average false positive rate across all classes.                            | Identifies model misclassification tendency.     |
| **Latency (ms/sample)** | Time cost per prediction sample.                                           | Measures computational efficiency.               |
| **Model Size (MB)**     | Disk size of the trained model.                                            | Reflects resource consumption and deployability. |

---

### 5. Experiment Workflow

#### **Q1: Binary vs Multiclass Comparison**

1. Load dataset.
2. Train Logistic Regression (LR) and Random Forest (RF).
3. Evaluate both in binary and multiclass tasks.
4. Record Accuracy, Macro-F1, Latency, and Model Size.

#### **Q2: Confusion Reduction Strategies**

1. Use multiclass dataset.
2. Apply:

   * **Class Weighting**
   * **SMOTE Oversampling**
   * **Threshold Adjustment**
3. Compare Macro-F1 and Macro-FPR across methods.

---

### 6. Key Findings

* Binary models achieved higher **Accuracy** and **F1-score**.
* Multiclass models suffered more **false positives** and lower stability.
* **SMOTE** and **threshold tuning** effectively improved class separation.
* **Class weighting** improved balance but increased misclassification for some classes.

---

### 7. How to Run

#### Install dependencies

```bash
python3 -m pip install -U pip
python3 -m pip install -U pandas scikit-learn imbalanced-learn joblib matplotlib python-pptx
```

#### Execute Q1

```bash
python3 q1_train_test.py --train "UNSW_NB15_training-set.csv" --test "UNSW_NB15_testing-set.csv"
```

#### Execute Q2

```bash
python3 q2_train_test.py --train "UNSW_NB15_training-set.csv" --test "UNSW_NB15_testing-set.csv"
```

#### Plot Results

```bash
python3 plot_results.py
```

---

### 8. Dataset

* **Dataset Used:** UNSW-NB15 (network intrusion detection dataset)
* For efficiency, about 10% of samples were extracted for comparison.
* Before running, unzip the datasets:

```bash
unzip -o "Processed_datasets.zip" -d data_zip1
unzip -o "Train_Test_datasets.zip" -d data_zip2
```

---

### 9. Future Work

* Extend to deep learning architectures (CNN, Transformer).
* Evaluate performance under different hardware (Edge vs Cloud).
* Integrate Explainable AI (SHAP, LIME) for interpretability.

---

### 10. License

This project is released under the MIT License.
Free for research, modification, and non-commercial use.

---



### 1. 项目简介

本项目研究在相同数据集、特征与计算资源条件下，**二分类与多分类模型的性能差异**，
并探索多种 **数据平衡与优化方法**（如类别权重、SMOTE、阈值调整）对多分类模型混淆度的影响。

研究问题包括：

1. **Q1：** 二分类与多分类模型在相同条件下性能差异如何？

   * 评价指标：Accuracy、Macro-F1、Macro-FPR、Latency（延迟/ms）、Model Size（模型大小/MB）

2. **Q2：** 使用 **Class Weights**、**SMOTE** 或 **Threshold Tuning** 是否能降低多分类混淆？

---

### 2. 项目结构

见英文版结构图。

---

### 3. 文件功能说明

| 文件名                  | 功能说明                                                           |
| -------------------- | -------------------------------------------------------------- |
| **q1_train_test.py** | 比较 Logistic Regression 与 Random Forest 在二分类与多分类下的性能差异。         |
| **q2_train_test.py** | 测试 Class Weight、SMOTE、Threshold 三种多分类优化策略。                     |
| **metrics_utils.py** | 定义 Accuracy、Precision、Recall、Macro-F1、Macro-FPR、Latency 等指标计算。 |
| **utils_data.py**    | 完成数据加载、清洗及训练测试集划分。                                             |
| **plot_results.py**  | 绘制性能对比图表，包括准确率、延迟与 F1 得分。                                      |
| **CSV / PNG 文件**     | 存储实验结果、混淆矩阵与可视化图像。                                             |

---

### 4. 指标说明

| 指标                      | 含义         | 用途             |
| ----------------------- | ---------- | -------------- |
| **Accuracy**            | 正确预测占比     | 衡量总体预测准确率      |
| **Macro-F1**            | 平均每类的 F1 值 | 衡量在不平衡数据中的综合表现 |
| **Macro-FPR**           | 平均误报率      | 反映模型的错误分类趋势    |
| **Latency (ms/sample)** | 每样本推理耗时    | 评估模型计算效率       |
| **Model Size (MB)**     | 模型文件大小     | 衡量部署与资源需求      |

---

### 5. 实验流程

#### Q1：二分类与多分类对比

1. 加载数据集
2. 训练 Logistic Regression 与 Random Forest
3. 分别在二分类和多分类任务中评估
4. 记录 Accuracy、Macro-F1、Latency、Model Size

#### Q2：多分类混淆优化

1. 使用相同数据集
2. 应用三种方法：

   * Class Weights
   * SMOTE 过采样
   * Threshold 阈值调整
3. 对比 Macro-F1 与 Macro-FPR 改进效果

---

### 6. 主要结论

* 二分类模型整体性能更高，尤其在 F1 与 Accuracy 上。
* 多分类模型的误报率和延迟更高。
* SMOTE 与 Threshold 调整对平衡性能有显著提升。
* Class Weight 可平衡类别但误报略增。

---

### 7. 运行方法

```bash
python3 -m pip install -U pandas scikit-learn imbalanced-learn joblib matplotlib python-pptx
python3 q1_train_test.py --train "UNSW_NB15_training-set.csv" --test "UNSW_NB15_testing-set.csv"
python3 q2_train_test.py --train "UNSW_NB15_training-set.csv" --test "UNSW_NB15_testing-set.csv"
python3 plot_results.py
```

---

### 8. 数据集

* 使用 UNSW-NB15 网络入侵检测数据集。
* 为提高效率，仅使用约 10% 的样本进行实验。
* 运行前需解压：

```bash
unzip -o "Processed_datasets.zip" -d data_zip1
unzip -o "Train_Test_datasets.zip" -d data_zip2
```

---

### 9. 后续拓展方向

* 尝试深度学习模型（CNN、Transformer）
* 进行边缘端与云端性能对比
* 应用可解释性分析（SHAP、LIME）提升可视化能力

---

### 10. 许可说明

本项目采用 **MIT License** 开源协议，可自由用于学习、研究与非商业用途。

---



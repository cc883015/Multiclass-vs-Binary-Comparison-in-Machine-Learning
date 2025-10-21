**dataset**UNSW-NB15: support zip，dataset  because the original dataset that is large，so maybe extract 10%.
 Binary vs Multiclass perfomance (Q1)
           Reduce Confusion  (Q2)

**question:**
question1.
 Under the same data, features, and computing power, what are the performance differences between binary and multi-class? 
 Metrics: **Macro-F1**, **Accuracy**, **Macro-FPR**, **idelay** (ms), **Model size**(MB) 

 question2.
 If **class weights** are changed, **SMOTE** is used, **threshold (confidence margin)** is changed, can the confusion of multiple classes be reduced?

 **dependency install** 
python3 -m pip install -U pip
python3 -m pip install -U scikit-learn
python3 -m pip install -U pandas scikit-learn imbalanced-learn joblib
python3 -m pip install -U pandas scikit-learn imbalanced-learn joblib matplotlib python-pptx

 **unzip dataset** 
 unzip -o "Processed_datasets.zip" -d data_zip1
 unzip -o "Train_Test_datasets.zip" -d data_zip2

**execute command**
python3 -m pip install -U pandas scikit-learn imbalanced-learn joblib matplotlib python-pptx
python3 -m pip install -U matplotlib

# q1
python3 q1_train_test.py \
  --train "UNSW_NB15_training-set.csv" \
  --test  "UNSW_NB15_testing-set.csv"


# q2

python3 q2_train_test.py --train "UNSW_NB15_training-set.csv" --test "UNSW_NB15_testing-set.csv" 

# ppt
python3 make_ppt.py --q1 "metrics_q1.csv" --q2 "metrics_q2.csv" --out "UNSW_NB15_results.pptx"

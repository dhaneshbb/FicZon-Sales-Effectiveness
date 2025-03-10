import chardet
import csv
import pandas as pd

file_path = (r"D:\00.Workspace\00 Projects\~ 02 pendding\FicZon Inc\data\raw\ficzone.csv")
with open(file_path, "rb") as f:
    raw_data = f.read(100000)  
    result = chardet.detect(raw_data)
result

with open(file_path, "r", encoding="ISO-8859-1") as f:
    sample = f.read(2000)  
    sniffer = csv.Sniffer()
    delimiter = sniffer.sniff(sample).delimiter
delimiter

data = pd.read_csv(file_path, encoding="ISO-8859-1", delimiter=";")

from tabulate import tabulate
from IPython.display import display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_colwidth', None)

from insightfulpy.eda import *

import warnings
import logging
import time
import joblib
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats
from scipy.stats import (
    chi2_contingency, fisher_exact, pearsonr, spearmanr, ttest_ind, 
    mannwhitneyu, shapiro
)
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, learning_curve, 
    StratifiedKFold, RandomizedSearchCV
)
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, OrdinalEncoder
)
from sklearn.pipeline import Pipeline
from sklearn.utils import resample, compute_class_weight, class_weight
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, balanced_accuracy_score, matthews_corrcoef, log_loss, 
    brier_score_loss, cohen_kappa_score, precision_recall_curve, 
    roc_curve, auc, classification_report, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve
import scikitplot as skplt
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMClassifier, LGBMRegressor
import catboost as cb
from catboost import CatBoostClassifier, CatBoostRegressor

import psutil
import os
import gc

def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB")

def dataframe_memory_usage(df):
    mem_usage = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"DataFrame Memory Usage: {mem_usage:.2f} MB")
    return mem_usage

def garbage_collection():
    gc.collect()
    memory_usage()

if __name__ == "__main__":
    memory_usage()

dataframe_memory_usage(data)

def normality_test_with_skew_kurt(df):
    normal_cols = []
    not_normal_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        col_data = df[col].dropna()
        if len(col_data) >= 3:
            if len(col_data) <= 5000:
                stat, p_value = shapiro(col_data)
                test_used = 'Shapiro-Wilk'
            else:
                stat, p_value = kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                test_used = 'Kolmogorov-Smirnov'
            col_skewness = skew(col_data)
            col_kurtosis = kurtosis(col_data)
            result = {
                'Column': col,
                'Test': test_used,
                'Statistic': stat,
                'p_value': p_value,
                'Skewness': col_skewness,
                'Kurtosis': col_kurtosis
            }
            if p_value > 0.05:
                normal_cols.append(result)
            else:
                not_normal_cols.append(result)
    normal_df = (
        pd.DataFrame(normal_cols)
        .sort_values(by='Column') 
        if normal_cols else pd.DataFrame(columns=['Column', 'Test', 'Statistic', 'p_value', 'Skewness', 'Kurtosis'])
    )
    not_normal_df = (
        pd.DataFrame(not_normal_cols)
        .sort_values(by='p_value', ascending=False)  # Sort by p-value descending (near normal to not normal)
        if not_normal_cols else pd.DataFrame(columns=['Column', 'Test', 'Statistic', 'p_value', 'Skewness', 'Kurtosis'])
    )
    print("\nNormal Columns (p > 0.05):")
    display(normal_df)
    print("\nNot Normal Columns (p ≤ 0.05) - Sorted from Near Normal to Not Normal:")
    display(not_normal_df)
    return normal_df, not_normal_df

def spearman_correlation(data, non_normal_cols, exclude_target=None, multicollinearity_threshold=0.8):
    if non_normal_cols.empty:
        print("\nNo non-normally distributed numerical columns found. Exiting Spearman Correlation.")
        return
    selected_columns = non_normal_cols['Column'].tolist()
    if exclude_target and exclude_target in selected_columns and pd.api.types.is_numeric_dtype(data[exclude_target]):
        selected_columns.remove(exclude_target)
    spearman_corr_matrix = data[selected_columns].corr(method='spearman')
    multicollinear_pairs = []
    for i, col1 in enumerate(selected_columns):
        for col2 in selected_columns[i+1:]:
            coef = spearman_corr_matrix.loc[col1, col2]
            if abs(coef) > multicollinearity_threshold:
                multicollinear_pairs.append((col1, col2, coef))
    print("\nVariables Exhibiting Multicollinearity (|Correlation| > {:.2f}):".format(multicollinearity_threshold))
    if multicollinear_pairs:
        for col1, col2, coef in multicollinear_pairs:
            print(f"- {col1} & {col2}: Correlation={coef:.4f}")
    else:
        print("No multicollinear pairs found.")
    annot_matrix = spearman_corr_matrix.round(2).astype(str)
    num_vars = len(selected_columns)
    fig_size = max(min(24, num_vars * 1.2), 10)  # Keep reasonable bounds
    annot_font_size = max(min(10, 200 / num_vars), 6)  # Smaller font for more variables
    plt.figure(figsize=(fig_size, fig_size * 0.75))
    sns.heatmap(
        spearman_corr_matrix,
        annot=annot_matrix,
        fmt='',
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        annot_kws={"size": annot_font_size},
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Spearman Correlation Matrix', fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.show()

def calculate_vif(data, exclude_target='TARGET', multicollinearity_threshold=5.0):
    # Select only numeric columns, exclude target, and drop rows with missing values
    numeric_data = data.select_dtypes(include=[np.number]).drop(columns=[exclude_target], errors='ignore').dropna()
    vif_data = pd.DataFrame()
    vif_data['Feature'] = numeric_data.columns
    vif_data['VIF'] = [variance_inflation_factor(numeric_data.values, i) 
                       for i in range(numeric_data.shape[1])]
    vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)
    high_vif = vif_data[vif_data['VIF'] > multicollinearity_threshold]
    low_vif = vif_data[vif_data['VIF'] <= multicollinearity_threshold]
    print(f"\nVariance Inflation Factor (VIF) Scores (multicollinearity_threshold = {multicollinearity_threshold}):")
    print("\nFeatures with VIF > threshold (High Multicollinearity):")
    if not high_vif.empty:
        print(high_vif.to_string(index=False))
    else:
        print("None. No features exceed the VIF threshold.")
    print("\nFeatures with VIF <= threshold (Low/No Multicollinearity):")
    if not low_vif.empty:
        print(low_vif.to_string(index=False))
    else:
        print("None. All features exceed the VIF threshold.")
    return vif_data, high_vif['Feature'].tolist()

print(data.shape)
for idx, col in enumerate(data.columns):
        print(f"{idx}: {col}")

data.head().T

detect_mixed_data_types(data)

cat_high_cardinality(data)

missing_inf_values(data)
print(f"\nNumber of duplicate rows: {data.duplicated().sum()}\n")
duplicates = data[data.duplicated()]
duplicates
show_missing(data)

data = data.drop_duplicates()

inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
print(f"Total Inf values: {inf_counts}")

data.dtypes.value_counts()

columns_info("Dataset Overview", data)

data.drop(columns=["Mobile", "Sales_Agent"], inplace=True)

data["Product_ID"].fillna(-1, inplace=True) 

data["Location"].fillna("Unknown", inplace=True)

data["Source"].fillna("Unknown", inplace=True)

missing_inf_values(data)

data["Product_ID"] = data["Product_ID"].astype(int)
data["Created"] = pd.to_datetime(data["Created"], errors="coerce")
data.drop(columns=["EMAIL"], inplace=True)

columns_info("Dataset Overview", data)

data["Created_Date"] = data["Created"].dt.date
data["Created_Hour"] = data["Created"].dt.hour
data["Created_DayOfWeek"] = data["Created"].dt.day_name()
data["Created_Month"] = data["Created"].dt.month_name()

analyze_data(data)

high_potential = ["CONVERTED", "converted", "Potential", "In Progress Positive", "Long Term"]
low_potential = ["Junk Lead", "Not Responding", "LOST", "In Progress Negative", "Just Enquiry", "Open"]
data["Lead_Category"] = data["Status"].apply(lambda x: 1 if x in high_potential else 0)

data_cat_missing_summary, data_cat_non_missing_summary = comp_cat_analysis(data, missing_df=True)
data_num_missing_summary, data_num_non_missing_summary = comp_num_analysis(data, missing_df=True)
data_outlier_summary, data_non_outlier_summary = comp_num_analysis(data, outlier_df=True)
print(data_cat_missing_summary.shape)
print(data_num_missing_summary.shape)
print(data_outlier_summary.shape)

data_cat_missing_summary

data_num_missing_summary

data_cat_non_missing_summary

data_num_non_missing_summary

data_non_outlier_summary

data_outlier_summary

data_negative_values = data.select_dtypes(include=[np.number]).lt(0).sum()
data_negative_values = data_negative_values[data_negative_values > 0].sort_values(ascending=False)
print("Columns with Negative Values (Sorted):\n", data_negative_values)

data_normal_df, data_not_normal_df = normality_test_with_skew_kurt(data)

cat_analyze_and_plot(data, 'Lead_Category')

grouped_summary(data, groupby='Lead_Category')

columns_info("Dataset Overview", data)

num_summary(data)

cat_summary(data)

kde_batches(data, batch_num=1)
box_plot_batches(data, batch_num=1)
qq_plot_batches(data, batch_num=1)

cat_bar_batches(data, batch_num=1)
cat_pie_chart_batches(data, batch_num=1, high_cardinality_limit=26)

num_vs_num_scatterplot_pair_batch(data, pair_num=0, batch_num=1, hue_column="Lead_Category")

num_vs_cat_box_violin_pair_batch(data, pair_num=0, batch_num=1, high_cardinality_limit=26, show_high_cardinality=False)

num_vs_cat_box_violin_pair_batch(data, pair_num=1, batch_num=1, high_cardinality_limit=26, show_high_cardinality=False)

cat_vs_cat_pair_batch(data, pair_num=1, batch_num=1,high_cardinality_limit=26, show_high_cardinality=False)

plt.figure(figsize=(15, 12))
plt.subplot(2, 2, 1)
daily_counts = data.groupby("Created_Date")["Lead_Category"].count()
sns.lineplot(x=daily_counts.index, y=daily_counts.values, marker="o")
plt.xticks(rotation=45)
plt.title("Leads Over Time (Daily)")
plt.xlabel("Date")
plt.ylabel("Number of Leads")
plt.subplot(2, 2, 2)
sns.countplot(x="Created_Hour", hue="Lead_Category", data=data, palette="coolwarm")
plt.title("Lead Distribution by Hour of Day")
plt.xlabel("Hour of the Day")
plt.ylabel("Count")
plt.subplot(2, 2, 3)
sns.countplot(x="Created_DayOfWeek", hue="Lead_Category", data=data, order=[
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], palette="coolwarm")
plt.xticks(rotation=45)
plt.title("Lead Distribution by Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Count")
plt.subplot(2, 2, 4)
sns.countplot(x="Created_Month", hue="Lead_Category", data=data, palette="coolwarm", order=[
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
plt.xticks(rotation=45)
plt.title("Lead Distribution by Month")
plt.xlabel("Month")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

data["Created_Month"] = data["Created"].dt.month
monthly_counts = data["Created_Month"].value_counts().sort_index()
monthly_trend = data.groupby("Created_Month")["Lead_Category"].mean()
all_months = pd.Series(index=range(1, 13), dtype=float)
monthly_trend = all_months.combine_first(monthly_trend)
monthly_counts = all_months.combine_first(monthly_counts)
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(monthly_trend.index, monthly_trend.values, marker='o', linestyle='-', color='blue', label="Conversion Rate")
ax1.set_ylabel("Conversion Rate", color="blue")
ax1.set_xlabel("Month")
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax1.tick_params(axis="y", labelcolor="blue")
ax2 = ax1.twinx()
ax2.bar(monthly_counts.index, monthly_counts.values, alpha=0.4, color='red', label="Lead Count")
ax2.set_ylabel("Lead Count", color="red")
ax2.tick_params(axis="y", labelcolor="red")
plt.title("Seasonal Trend: Lead Conversion Rate & Lead Volume")
ax1.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

columns_info("Dataset Overview", data)

plot_boxplots(data)
calculate_skewness_kurtosis(data)

detect_outliers(data)

data_normal_df, data_not_normal_df = normality_test_with_skew_kurt(data)
spearman_correlation(data, data_not_normal_df, exclude_target="Lead_Category", multicollinearity_threshold=0.8)

at_above, at_below =calculate_vif(data, exclude_target='Lead_Category', multicollinearity_threshold=8.0)

data_copy = data.copy()
categorical_cols = ["Source", "Location", "Delivery_Mode", "Status", "Created_DayOfWeek"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data_copy[col] = le.fit_transform(data_copy[col])
    label_encoders[col] = le  # Save encoder for future use
numerical_cols = ["Product_ID", "Created_Hour", "Created_Month"]
categorical_cols = ["Source", "Location", "Delivery_Mode", "Status", "Created_DayOfWeek"]
mi_scores = mutual_info_classif(data_copy[categorical_cols], data_copy["Lead_Category"], discrete_features=True)
mi_df = pd.DataFrame({"Feature": categorical_cols, "MI Score": mi_scores})
X = data_copy[numerical_cols + categorical_cols]
y = data_copy["Lead_Category"]
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
combined_df = pd.concat([mi_df, feature_importances], axis=0).fillna(0)
combined_df = combined_df.sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 5))
sns.barplot(y=combined_df["Feature"], x=combined_df["Importance"], palette="coolwarm")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance for Classification Model")
plt.show()
print(combined_df)

columns_to_remove = ["Created", "Created_Date", "Status", "Created_DayOfWeek", "Created_Hour"]
data.drop(columns=columns_to_remove, inplace=True)

frequency_encoded_cols = ["Source", "Location"]
for col in frequency_encoded_cols:
    data[col + "_freq"] = data[col].map(data[col].value_counts())  
data.drop(columns=frequency_encoded_cols, inplace=True)
one_hot_encoded_cols = ["Delivery_Mode"]
data = pd.get_dummies(data, columns=one_hot_encoded_cols, drop_first=True)  

data["Lead_Category"] = data["Lead_Category"].astype("int8") 
data["Source_freq"] = data["Source_freq"].astype("int32") 
data["Location_freq"] = data["Location_freq"].astype("int32") 

X = data.drop(columns='Lead_Category')
y = data['Lead_Category']

print("Class distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

classes = np.array([0, 1])
class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
class_weight_dict

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Train and evaluate a model with additional metrics."""
    start_time = time.time()
    model.fit(X_train, y_train)  # Train the model
    train_time = time.time() - start_time  
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        log_loss_score = log_loss(y_test, y_proba)
        brier_score = brier_score_loss(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        log_loss_score, brier_score, roc_auc = None, None, None  
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=1)
    rec = recall_score(y_test, y_pred)
    f1_metric = f1_score(y_test, y_pred)
    acc_train = accuracy_score(y_train, y_pred_train)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    strat_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=strat_kfold, scoring='f1').mean()
    overfit = acc_train - acc
    return {
        "Training Time (seconds)": round(train_time, 3),
        "Accuracy": acc,
        "Balanced Accuracy": balanced_acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1_metric,
        "MCC (Matthews Correlation Coefficient)": mcc,
        "Cohen's Kappa": kappa,
        "ROC AUC Score": roc_auc,
        "Log Loss": log_loss_score,
        "Brier Score": brier_score,
        "Cross-Validation F1-Score": cv_f1,
        "True Negatives (TN)": tn,
        "False Positives (FP)": fp,
        "False Negatives (FN)": fn,
        "True Positives (TP)": tp,
        "Training Accuracy": acc_train,
        "Overfit (Train - Test Acc)": overfit
    }

def threshold_analysis(model, X_test, y_test, thresholds=np.arange(0.1, 1.0, 0.1)):
    y_probs = model.predict_proba(X_test)[:, 1]  
    results = []
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        results.append({
            "Threshold": round(threshold, 2),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "Accuracy": round(accuracy, 4),
            "True Negatives (TN)": tn,
            "False Positives (FP)": fp,
            "False Negatives (FN)": fn,
            "True Positives (TP)": tp
        })
    df_results = pd.DataFrame(results)
    best_threshold = df_results.loc[df_results["F1-Score"].idxmax(), "Threshold"]
    print(f" Best Decision Threshold (Max F1-Score): {best_threshold:.2f}")
    return df_results, best_threshold
    
def cross_validation_analysis_table(model, X_train, y_train, cv_folds=5, scoring_metric="f1"):
    strat_kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=strat_kfold, scoring=scoring_metric)
    cv_results_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(cv_folds)],
        "F1-Score": scores
    })
    cv_results_df.loc["Mean"] = ["Mean", np.mean(scores)]
    cv_results_df.loc["Std"] = ["Standard Deviation", np.std(scores)]
    return cv_results_df

def plot_all_evaluation_metrics(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
    y_pred_default = (y_probs >= 0.4).astype(int)
    cm = confusion_matrix(y_test, y_pred_default)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes[0, 0].plot(prob_pred, prob_true, marker="o", label="Calibration")
    axes[0, 0].plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated")
    axes[0, 0].set_title("Calibration Curve")
    axes[0, 0].set_xlabel("Predicted Probability")
    axes[0, 0].set_ylabel("Actual Probability")
    axes[0, 0].legend()
    axes[0, 0].grid()
    skplt.metrics.plot_cumulative_gain(y_test, model.predict_proba(X_test), ax=axes[0, 1])
    axes[0, 1].set_title("Cumulative Gains Curve")
    y_probs_1 = y_probs[y_test == 1]  # Positive class
    y_probs_0 = y_probs[y_test == 0]  # Negative class
    axes[0, 2].hist(y_probs_1, bins=50, alpha=0.5, label="y=1")
    axes[0, 2].hist(y_probs_0, bins=50, alpha=0.5, label="y=0")
    axes[0, 2].set_title("Kolmogorov-Smirnov (KS) Statistic")
    axes[0, 2].set_xlabel("Predicted Probability")
    axes[0, 2].set_ylabel("Frequency")
    axes[0, 2].legend()
    axes[0, 2].grid()
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = np.linspace(0.6, 0.9, 10)
    val_scores = np.linspace(0.55, 0.85, 10)
    axes[1, 0].plot(train_sizes, train_scores, label="Train Score")
    axes[1, 0].plot(train_sizes, val_scores, label="Validation Score")
    axes[1, 0].set_title("Learning Curve (Simulated)")
    axes[1, 0].set_xlabel("Training Size")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend()
    axes[1, 0].grid()
    skplt.metrics.plot_lift_curve(y_test, model.predict_proba(X_test), ax=axes[1, 1])
    axes[1, 1].set_title("Lift Curve")
    axes[1, 2].plot(thresholds, precision[:-1], "b--", label="Precision")
    axes[1, 2].plot(thresholds, recall[:-1], "r-", label="Recall")
    axes[1, 2].set_title("Precision-Recall Curve")
    axes[1, 2].set_xlabel("Threshold")
    axes[1, 2].set_ylabel("Score")
    axes[1, 2].legend()
    axes[1, 2].grid()
    axes[2, 0].plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    axes[2, 0].plot([0, 1], [0, 1], linestyle="--", color="black")
    axes[2, 0].set_title("ROC Curve")
    axes[2, 0].set_xlabel("False Positive Rate")
    axes[2, 0].set_ylabel("True Positive Rate")
    axes[2, 0].legend()
    axes[2, 0].grid()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[2, 1], cmap="Blues")
    axes[2, 1].set_title("Confusion Matrix")
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm)
    disp_norm.plot(ax=axes[2, 2], cmap="Blues")
    axes[2, 2].set_title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.show()

base_models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
    "SVM": SVC(class_weight="balanced", probability=True, random_state=42),  
    "XGBoost": XGBClassifier(scale_pos_weight=class_weight_dict[1], use_label_encoder=False, eval_metric="logloss", random_state=42),
    "LightGBM": LGBMClassifier(class_weight="balanced", random_state=42),
    "CatBoost": CatBoostClassifier(auto_class_weights="Balanced", verbose=0, random_state=42)  
}

model_results = {}
for model_name, model in base_models.items():
    print(f"Training & Evaluating: {model_name}...")
    results = evaluate_model(model, X_train, y_train, X_test, y_test)
    model_results[model_name] = results  
df_results = pd.DataFrame.from_dict(model_results, orient='index')
base_results = df_results.sort_values(by="Recall", ascending=False)

base_results

models = {
    "LightGBM": lgb.LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
    "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss")
}
param_grids = {
    "LightGBM": {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "num_leaves": [20, 31, 50],
        "min_child_samples": [10, 20, 30],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [0, 0.1, 0.5],
        "class_weight": ["balanced", None] 
    },
    "CatBoost": {
        "iterations": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [3, 5, 7],
        "l2_leaf_reg": [1, 3, 5],
        "border_count": [32, 64, 128],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bylevel": [0.7, 0.9, 1.0]
    },
    "XGBoost": {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.5],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [0, 0.1, 0.5],
        "scale_pos_weight": [1.0, 1.3023]  
    }
}
best_models = {}
for model_name, model in models.items():
    print(f"Tuning {model_name}...")
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grids[model_name],
        n_iter=30,  
        cv=5,  
        scoring="f1",  
        verbose=2,
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)
    best_models[model_name] = search.best_estimator_

best_models

evaluation_results = {}
for model_name, best_model in best_models.items():
    print(f"Evaluating {model_name}...")
    evaluation_results[model_name] = evaluate_model(best_model, X_train, y_train, X_test, y_test)
evaluation_df = pd.DataFrame(evaluation_results).T 

evaluation_df

lgbm_best = best_models["LightGBM"]
catboost_best = best_models["CatBoost"]
xgb_best = best_models["XGBoost"]
ensemble_model = VotingClassifier(
    estimators=[
        ("LightGBM", lgbm_best),
        ("CatBoost", catboost_best),
        ("XGBoost", xgb_best)
    ],
    voting="soft" 
)
ensemble_model.fit(X_train, y_train)
ensemble_results = evaluate_model(ensemble_model, X_train, y_train, X_test, y_test)
ensemble_df = pd.DataFrame([ensemble_results], index=["Ensemble (Voting)"])

ensemble_df

stacking_estimators = [
    ("LightGBM", best_models["LightGBM"]),
    ("CatBoost", best_models["CatBoost"]),
    ("XGBoost", best_models["XGBoost"])
    
]
meta_model = LogisticRegression(class_weight="balanced", random_state=42)
stacking_model = StackingClassifier(
    estimators=stacking_estimators,
    final_estimator=meta_model,  # Meta-learner
    passthrough=False,  
    cv=5, 
    n_jobs=-1
)
stacking_model.fit(X_train, y_train)
stacking_results = evaluate_model(stacking_model, X_train, y_train, X_test, y_test)
stacking_df = pd.DataFrame([stacking_results], index=["Stacking Ensemble"])

stacking_df

base_results["Source"] = "Base Models"
evaluation_df["Source"] = "Tuned Models"
ensemble_df["Source"] = "Voting Ensemble"
stacking_df["Source"] = "Stacking Ensemble"
comparison_df = pd.concat([base_results, evaluation_df, ensemble_df, stacking_df])
comparison_df = comparison_df.reset_index().rename(columns={"index": "Model"})
comparison_df = comparison_df[["Source", "Model"] + [col for col in comparison_df.columns if col not in ["Source", "Model"]]]
comparison_df

def print_feature_importance_xgb(model, model_name, feature_names):
    """Print feature importance for XGBoost model in tabular format."""
    if hasattr(model, "feature_importances_"): 
        importances = model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]  
        table = [[feature_names[idx], f"{importances[idx]:.5f}"] for idx in sorted_indices]
        print(f"\n Feature Importance - {model_name}")
        print(tabulate(table, headers=["Feature", "Importance"], tablefmt="pipe"))
    else:
        print(f" {model_name} does not support feature importance.")
feature_names = X_train.columns.tolist()
print_feature_importance_xgb(best_models["XGBoost"], "XGBoost", feature_names)

final_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=3,
    colsample_bytree=0.7,
    gamma=0,
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=class_weight_dict[1]  
)
final_model.fit(X_train, y_train)

cv_results_table = cross_validation_analysis_table(final_model, X_train, y_train, cv_folds=5, scoring_metric="f1")
cv_results_table

df_threshold_results, best_threshold = threshold_analysis(final_model, X_test, y_test)
df_threshold_results = df_threshold_results.sort_values(by="F1-Score", ascending=False)
print("\n=== Threshold Comparison Table ===")
df_threshold_results

if not hasattr(final_model, "predict_proba"):
    raise ValueError(" Model does not support probability predictions.")
y_probs = final_model.predict_proba(X_test)[:, 1]
y_pred_final = (y_probs >= 0.40).astype(int)
print("\n Final Classification Report (Final Adjusted Threshold - 0.40):")
print(classification_report(y_test, y_pred_final))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_final).ravel()
print(f"\nConfusion Matrix at Threshold 0.40:")
print(f"True Negatives (TN): {tn}, False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}, True Positives (TP): {tp}")

plot_all_evaluation_metrics(final_model, X_test, y_test)

joblib.dump(final_model, "final_xgb_model.pkl")
print("Model saved successfully!")

final_model = joblib.load("final_xgb_model.pkl")
y_probs = final_model.predict_proba(X_test)[:, 1]
y_pred_final = (y_probs >= 0.40).astype(int)
print("\n LightGBM Classification Report (Final Adjusted Threshold - 0.40):")
print(classification_report(y_test, y_pred_final))

import types
import inspect

user_funcs = [name for name in globals() if isinstance(globals()[name], types.FunctionType) and globals()[name].__module__ == '__main__']
imported_funcs = [name for name in globals() if isinstance(globals()[name], types.FunctionType) and globals()[name].__module__ != '__main__']
imported_pkgs = [name for name in globals() if isinstance(globals()[name], types.ModuleType)]
print("Imported packages:")
for i, alias in enumerate(imported_pkgs, 1):
    print(f"{i}: {globals()[alias].__name__}")
print("\nUser-defined functions:")
for i, func in enumerate(user_funcs, 1):
    print(f"{i}: {func}")
print("\nImported functions:")
for i, func in enumerate(imported_funcs, 1):
    print(f"{i}: {func}")
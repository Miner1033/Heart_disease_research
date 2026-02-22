"""
Chronic Kidney Disease (CKD) Stage Classification Using Ensemble Machine Learning
Research Code for ICEFronT 2026 - MBSTU
Dataset: CKD with Stages (Kaggle - aryannandanwar)
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, roc_auc_score,
                             precision_score, recall_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 2. LOAD DATA
# ============================================================
# Download from: https://www.kaggle.com/datasets/aryannandanwar/ckdchronic-kidney-disease-dataset-with-stages
df = pd.read_csv("ckd_dataset_with_stages.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nColumn Names:\n{df.columns.tolist()}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# ============================================================
# 3. PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("PREPROCESSING")
print("=" * 60)

# Auto-detect target column (handles 'Stage', 'ckd_stage', 'CKD_Stage', etc.)
possible_targets = ['ckd_stage', 'Stage', 'stage', 'CKD_Stage', 'target', 'class']
TARGET_COL = next((c for c in possible_targets if c in df.columns), None)
if TARGET_COL is None:
    # Fallback: use last column
    TARGET_COL = df.columns[-1]
print(f"Target column detected: '{TARGET_COL}'")
print(f"\nClass Distribution:\n{df[TARGET_COL].value_counts()}")

# Separate features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Encode categorical columns
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode target
target_le = LabelEncoder()
y = target_le.fit_transform(y)
print(f"Classes: {target_le.classes_}")

# Impute missing values
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
print(f"After SMOTE - X shape: {X_balanced.shape}")
print(f"Class distribution after SMOTE: {np.bincount(y_balanced)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# ============================================================
# 4. MODEL TRAINING & EVALUATION
# ============================================================
print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

models = {
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
    "XGBoost":             XGBClassifier(n_estimators=200, random_state=42, eval_metric='mlogloss'),
    "SVM":                 SVC(kernel='rbf', probability=True, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

results = {}
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    # 10-fold cross-validation on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec  = recall_score(y_test, y_pred, average='weighted')
    f1   = f1_score(y_test, y_pred, average='weighted')

    results[name] = {
        "CV Accuracy (Mean)": round(cv_scores.mean() * 100, 2),
        "CV Std":             round(cv_scores.std() * 100, 2),
        "Test Accuracy":      round(acc * 100, 2),
        "Precision":          round(prec * 100, 2),
        "Recall":             round(rec * 100, 2),
        "F1-Score":           round(f1 * 100, 2),
    }
    print(f"\n{name}:")
    print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"  Test Accuracy: {acc*100:.2f}%  F1: {f1*100:.2f}%")

results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
print("\n" + results_df.to_string(index=False))

# ============================================================
# 5. BEST MODEL DETAILED ANALYSIS (XGBoost)
# ============================================================
print("\n" + "=" * 60)
print("BEST MODEL: XGBoost - DETAILED REPORT")
print("=" * 60)

best_model = models["XGBoost"]
y_pred_best = best_model.predict(X_test)
target_names_str = [str(c) for c in target_le.classes_]
print(classification_report(y_test, y_pred_best, target_names=target_names_str))

# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================
importances = best_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feat_imp_df.head(10).to_string(index=False))

# ============================================================
# 7. VISUALIZATIONS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("CKD Stage Classification - Analysis Results", fontsize=16, fontweight='bold')

# Plot 1: Accuracy Comparison
ax1 = axes[0, 0]
bars = ax1.barh(results_df['Model'], results_df['Test Accuracy'],
                color=['#2196F3','#4CAF50','#FF9800','#F44336','#9C27B0','#00BCD4','#FF5722'])
ax1.set_xlabel("Accuracy (%)")
ax1.set_title("Model Accuracy Comparison")
ax1.set_xlim([80, 101])
for bar, val in zip(bars, results_df['Test Accuracy']):
    ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9)

# Plot 2: Confusion Matrix
ax2 = axes[0, 1]
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=target_names_str, yticklabels=target_names_str)
ax2.set_title("XGBoost Confusion Matrix")
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")

# Plot 3: Feature Importance
ax3 = axes[1, 0]
top_feat = feat_imp_df.head(10)
ax3.barh(top_feat['Feature'], top_feat['Importance'], color='#4CAF50')
ax3.set_title("Top 10 Feature Importances (XGBoost)")
ax3.set_xlabel("Importance Score")
ax3.invert_yaxis()

# Plot 4: F1-Score Comparison
ax4 = axes[1, 1]
ax4.bar(results_df['Model'], results_df['F1-Score'],
        color=['#2196F3','#4CAF50','#FF9800','#F44336','#9C27B0','#00BCD4','#FF5722'])
ax4.set_title("F1-Score Comparison")
ax4.set_ylabel("F1-Score (%)")
ax4.set_ylim([80, 101])
ax4.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig("ckd_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as ckd_results.png")

# ============================================================
# 8. ENSEMBLE (VOTING) MODEL
# ============================================================
print("\n" + "=" * 60)
print("ENSEMBLE VOTING CLASSIFIER")
print("=" * 60)

ensemble = VotingClassifier(estimators=[
    ('rf',  RandomForestClassifier(n_estimators=200, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=200, random_state=42, eval_metric='mlogloss')),
    ('gb',  GradientBoostingClassifier(n_estimators=200, random_state=42)),
], voting='soft')

ensemble.fit(X_train, y_train)
y_pred_ens = ensemble.predict(X_test)
print(f"Ensemble Accuracy: {accuracy_score(y_test, y_pred_ens)*100:.2f}%")
print(f"Ensemble F1-Score: {f1_score(y_test, y_pred_ens, average='weighted')*100:.2f}%")
print("\n" + classification_report(y_test, y_pred_ens, target_names=target_names_str))

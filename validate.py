import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import r2_score, mean_absolute_error, classification_report, f1_score

FILE_NAME = "human_digital_behavior_wellbeing.csv"

def validate_dataset(filepath):
    print("=" * 70)
    print(" 1. LOADING & DATASET STRUCTURE VALIDATION")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # 1. Missing Values Check
    missing_vals = df.isnull().sum().sum()
    print(f"Missing Values: {missing_vals}")
    assert missing_vals == 0, "Validation Failed: Dataset contains missing values!"

    # 2. Duplicate IDs Check
    unique_ids = df['user_id'].nunique()
    print(f"Unique User IDs: {unique_ids:,} / {len(df):,}")
    assert unique_ids == len(df), "Validation Failed: Duplicate user_ids found!"

    print("\n" + "=" * 70)
    print(" 2. TARGET DISTRIBUTIONS & SANITY CHECKS")
    print("=" * 70)
    
    # Continuous Target Check
    score = df['wellbeing_score']
    print("Continuous Target ('wellbeing_score'):")
    print(f"  Min: {score.min():.2f} | Max: {score.max():.2f} | Mean: {score.mean():.2f} | Std: {score.std():.2f}")
    assert score.min() >= 0 and score.max() <= 100, "Validation Failed: Score out of bounds [0, 100]!"

    # Categorical Target Balance Check
    print("\nCategorical Target ('wellbeing_category') Class Balance:")
    cat_dist = df['wellbeing_category'].value_counts(normalize=True) * 100
    for category, val in cat_dist.items():
        print(f"  - {category:<10}: {val:.2f}%")

    print("\n" + "=" * 70)
    print(" 3. TOP FEATURE CORRELATIONS WITH TARGET")
    print("=" * 70)
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    corrs = df[num_cols].corr()['wellbeing_score'].drop('wellbeing_score').sort_values()
    
    print("Top Negative Correlations:")
    for col, corr in corrs.head(5).items():
        print(f"  - {col:<30}: {corr:.3f}")
        
    print("\nTop Positive Correlations:")
    for col, corr in corrs.tail(5).items():
        print(f"  - {col:<30}: {corr:.3f}")

    print("\n" + "=" * 70)
    print(" 4. BASELINE MODEL BENCHMARK (REGRESSION & CLASSIFICATION)")
    print("=" * 70)
    
    # Prepare features
    drop_cols = ['user_id', 'wellbeing_score', 'wellbeing_category', 'first_screen_time', 'last_screen_time']
    X = df.drop(columns=drop_cols)
    X = pd.get_dummies(X, drop_first=True) # One-hot encode categoricals
    
    y_reg = df['wellbeing_score']
    y_cls = df['wellbeing_category']

    # Train / Test split on a 20,000 row sample for speed during validation
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X[:20000], y_reg[:20000], test_size=0.2, random_state=42
    )
    _, _, y_cls_train, y_cls_test = train_test_split(
        X[:20000], y_cls[:20000], test_size=0.2, random_state=42
    )

    # A. Regression Baseline
    reg = HistGradientBoostingRegressor(random_state=42)
    reg.fit(X_train, y_reg_train)
    reg_preds = reg.predict(X_test)
    r2 = r2_score(y_reg_test, reg_preds)
    mae = mean_absolute_error(y_reg_test, reg_preds)
    
    print(f"Regression Task Baseline (HistGradientBoosting):")
    print(f"  R² Score: {r2:.4f}")
    print(f"  MAE     : {mae:.4f}")
    
    # Assert ideal Kaggle dataset properties (Predictable, but with non-trivial noise/complexity)
    assert 0.50 <= r2 <= 0.92, f"Warning: R² of {r2:.2f} indicates target is either too random or too easy!"

    # B. Classification Baseline
    clf = HistGradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_cls_train)
    cls_preds = clf.predict(X_test)
    macro_f1 = f1_score(y_cls_test, cls_preds, average='macro')
    
    print(f"\nClassification Task Baseline (HistGradientBoosting):")
    print(f"  Macro F1-Score: {macro_f1:.4f}")

    print("\n" + "=" * 70)
    print(" PASSED: Dataset structure & signal-to-noise ratio are Kaggle-ready!")
    print("=" * 70)

if __name__ == "__main__":
    validate_dataset(FILE_NAME)
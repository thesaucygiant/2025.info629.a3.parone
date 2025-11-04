"""
Customer Churn Prediction - Supervised Learning Demo
Binary Classification using Random Forest

Dataset: Telco Customer Churn from Kaggle
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Package Requirements:

pandas 2.3.2
numpy 2.2.6
scikit-learn 1.7.2
matplotlib 3.10.7
seaborn 0.13.2
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, classification_report
)
from sklearn.tree import plot_tree, export_text
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
def load_data():
    #Load the Telco Customer Churn dataset
    #Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    
    # For demo purposes, using publicly available dataset
    #url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    #WA_Fn-UseC_-Telco-Customer-Churn
    #df = pd.read_csv(url)
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    
    
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")
    
    return df

def preprocess_data(df):
    #Clean and prepare data for modeling
    #Make a copy to avoid warnings
    df = df.copy()
    
    # Convert TotalCharges to numeric and fill missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Drop customerID
    df = df.drop('customerID', axis=1)
    
    # Convert target variable
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    
    print(f"\nData preprocessed. Shape: {df.shape}")
    return df

def train_model(X_train, y_train):
    """Train Random Forest Classifier"""
    print("\nTraining Random Forest Classifier...")
    
    # Initialize model with balanced class weights
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    
    print(f"\n{'CONFUSION MATRIX':^40}")
    print(f"{'':20} {'Predicted':^20}")
    print(f"{'':20} {'No Churn':^10} {'Churn':^10}")
    print(f"{'Actual No Churn':20} {tn:^10} {fp:^10}")
    print(f"{'Actual Churn':20} {fn:^10} {tp:^10}")
    
    print(f"\nTrue Negatives (TN):  {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP):  {tp}")
    
    print("\n" + classification_report(y_test, y_pred, 
                                       target_names=['No Churn', 'Churn']))
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
    }

def feature_importance(model, feature_names):
    """Display feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    print("\n" + "="*60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*60)
    for i, idx in enumerate(indices, 1):
        print(f"{i:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(10), importances[indices])
    plt.xticks(range(10), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("\nFeature importance plot saved as 'feature_importance.png'")

def test_cases(model, feature_names):
    """Test with specific examples"""
    print("\n" + "="*60)
    print("TEST PREDICTIONS")
    print("="*60)
    
    # Test Case 1
    testcase1 = {
        'gender': 1, 'SeniorCitizen': 0, 'Partner': 0, 'Dependents': 0,
        'tenure': 3, 'PhoneService': 1, 'MultipleLines': 0, 
        'InternetService': 1, 'OnlineSecurity': 0, 'OnlineBackup': 0,
        'DeviceProtection': 0, 'TechSupport': 0, 'StreamingTV': 0,
        'StreamingMovies': 0, 'Contract': 0, 'PaperlessBilling': 1,
        'PaymentMethod': 2, 'MonthlyCharges': 95.0, 'TotalCharges': 285.0
    }
    
    # Test Case 2
    testcase2 = {
        'gender': 0, 'SeniorCitizen': 1, 'Partner': 1, 'Dependents': 1,
        'tenure': 60, 'PhoneService': 1, 'MultipleLines': 1,
        'InternetService': 0, 'OnlineSecurity': 2, 'OnlineBackup': 1,
        'DeviceProtection': 1, 'TechSupport': 1, 'StreamingTV': 1,
        'StreamingMovies': 1, 'Contract': 2, 'PaperlessBilling': 0,
        'PaymentMethod': 0, 'MonthlyCharges': 55.0, 'TotalCharges': 3300.0
    }
    
    # Test Case 3
    testcase3 = {
        'gender': 1, 'SeniorCitizen': 0, 'Partner': 1, 'Dependents': 0,
        'tenure': 18, 'PhoneService': 1, 'MultipleLines': 1,
        'InternetService': 1, 'OnlineSecurity': 0, 'OnlineBackup': 1,
        'DeviceProtection': 1, 'TechSupport': 1, 'StreamingTV': 0,
        'StreamingMovies': 1, 'Contract': 1, 'PaperlessBilling': 1,
        'PaymentMethod': 1, 'MonthlyCharges': 70.0, 'TotalCharges': 1260.0
    }
    
    # Test Case 4
    testcase4 = {
        'gender': 1, 'SeniorCitizen': 0, 'Partner': 1, 'Dependents': 8,
        'tenure': 18, 'PhoneService': 0, 'MultipleLines': 0,
        'InternetService': 1, 'OnlineSecurity': 0, 'OnlineBackup': 0,
        'DeviceProtection': 0, 'TechSupport': 0, 'StreamingTV': 1,
        'StreamingMovies': 1, 'Contract': 1, 'PaperlessBilling': 1,
        'PaymentMethod': 1, 'MonthlyCharges': 200.0, 'TotalCharges': 1260.0
    }
    
    testcases = [
        ("Test Case 1: New customer, expensive, month-to-month, no support", testcase1),
        ("Test Case 2: Long tenure, 2-year contract, full services", testcase2),
        ("Test Case 3: Moderate tenure, 1-year contract, mixed services", testcase3),
        ("Test Case 4: Moderate tenure, 8 dependents, 1-year contract, mixed services", testcase4)
    ]
    
    for desc, testcase in testcases:
        X_testcase = pd.DataFrame([testcase])[feature_names]
        pred = model.predict(X_testcase)[0]
        proba = model.predict_proba(X_testcase)[0]
        
        print(f"\n{desc}")
        print(f"  Prediction: {'CHURN' if pred == 1 else 'NO CHURN'}")
        print(f"  Probability: {proba[1]*100:.1f}% chance of churn")
        
def visualize_single_tree(model, feature_names, tree_index=0):
    #Visualize a single decision tree from the Random Forest model
    
    # Get a single tree from the forest
    single_tree = model.estimators_[tree_index]
    
    # Create visualization
    plt.figure(figsize=(20, 10))
    plot_tree(single_tree, 
              feature_names=feature_names,
              class_names=['No Churn', 'Churn'],
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title(f'Decision Tree #{tree_index} from Random Forest')
    plt.tight_layout()
    plt.savefig(f'decision_tree_{tree_index}.png', dpi=300, bbox_inches='tight')
    print(f"\nDecision tree #{tree_index} saved as 'decision_tree_{tree_index}.png'")
    
    # Also export text representation
    tree_rules = export_text(single_tree, feature_names=feature_names)
    with open(f'tree_rules_{tree_index}.txt', 'w') as f:
        f.write(tree_rules)
    print(f"Tree rules saved as 'tree_rules_{tree_index}.txt'")


def main():
    """Starting the analysis"""
    print("="*60)
    print("Course: INFO-629-686 - FA 25-26")
    print("Assignment 3: SUPERVISED LEARNING DEMO - CUSTOMER CHURN PREDICTION")
    print("Student: Anthony Parone")
    print("Date: November 2025")
    print("="*60)
    
    # Load the data into a dataframe
    df = load_data()
    
    #2 Preprocess - this cleans up the data for the model training
    df = preprocess_data(df)
    
    #3 Split features and target
    #remove the column churn and leave all the features for X
    X = df.drop('Churn', axis=1)
    #only the churn values get assigned to Y
    y = df['Churn']
    
    print(f"\nClass distribution:")
    print(f"  No Churn: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"  Churn:    {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
    
    # Train-test split - The Split Creates 4 Sets:
    #`X_train`** - Features for training (70% of data)
    #**`X_test`** - Features for testing (30% of data)
    # **`y_train`** - Target labels for training (70% of data)
    #**`y_test`** - Target labels for testing (30% of data)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    
    # Train model
    model = train_model(X_train, y_train)
    #vizualize one tree
    visualize_single_tree(model, X.columns.tolist(), tree_index=0)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    
    # Feature importance
    feature_importance(model, X.columns.tolist())
    
    # Test examples
    test_cases(model, X.columns.tolist())
    
    #print("\n" + "="*60)
    #print("DEMO COMPLETED SUCCESSFULLY")
    #print("="*60)

if __name__ == "__main__":
    main()


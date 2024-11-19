## Data Preprocessing
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
data =pd.read_csv(r"C:\Users\praga\Downloads\Data_Cortex_Nuclear.csv")
data.head()
data.info()
data.describe(include='all')
data.isnull().sum()
**Identifying Missing Values**
missing_values=data.isnull().sum()
missing_values[missing_values>0]

# Droping columns with more than 20% missing values
threshold = 0.1 * len(data)
data = data.dropna(thresh=threshold, axis=1)
**Imputing missing values using meadian strategy**
numeric_columns = data.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='median')
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
print(data.info())
**Normalization/Standardization**
numeric_columns = data.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
**Encoding Categorical Variables**
scaler = StandardScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
cols=[ 'Genotype', 'Treatment', 'Behavior', 'class']
le=LabelEncoder()

for i in cols:
    data[i]=le.fit_transform(data[i])

print(data.head())
print(data.info())

data.to_csv('clean_data.csv', index=False)
## EDA
data=pd.read_csv('clean_data.csv')
print(data.describe())
print(data.describe(include=['object']))
print(data.isnull().sum())
**Verification of any missing values**
plt.figure(figsize=(10, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()
**Plotting histograms for each protiens**
proteins = data.columns[1:-4]  
plt.figure(figsize=(20, 100))
for i, protein in enumerate(proteins, 1):
    plt.subplot(len(proteins) // 4 + 1, 4, i)
    plt.hist(data[protein], bins=30, alpha=0.7, color='b', edgecolor='black')
    plt.title(f'Histogram of {protein}')
    plt.xlabel(protein)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
**Plotting Box plot for outliers**
plt.figure(figsize=(20, 100))
for i, protein in enumerate(proteins, 1):
    plt.subplot(len(proteins) // 4 + 1, 4, i)
    plt.boxplot(data[protein], vert=False)
    plt.title(f'Box Plot of {protein}')
    plt.xlabel(protein)
plt.tight_layout()
plt.show()
**Plotting all Box plot for outliers in one graph**
plt.figure(figsize=(20, 20))
sns.boxplot(data=data, orient="h")
plt.show()
**Correlation matrix**
correlation_matrix = data[proteins].corr()
print(correlation_matrix)
**Heatmap for correlation matrix**
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
**Scatter plot for outliers**
#Calculating IQR(InterQuartile Range) for finding outliers & ploting it using scatter plots
plt.figure(figsize=(40, 120))
for i, protein in enumerate(proteins, 1):
    Q1 = data[protein].quantile(0.25)
    Q3 = data[protein].quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data[protein] < (Q1 - 1.5 * IQR)) | (data[protein] > (Q3 + 1.5 * IQR))]
    plt.subplot(len(proteins) // 4 + 1, 4, i)
    plt.scatter(data.index, data[protein], alpha=0.5, label='Data')
    plt.scatter(outliers.index, outliers[protein], color='r', label='Outliers')
    plt.title(f'Scatter Plot of {protein}', fontsize=20)
    plt.xlabel('Index', fontsize=15)
    plt.ylabel(protein, fontsize=15)
    plt.legend()
plt.tight_layout()
plt.show()
**Count plots for categorical variables**
categorical_columns = ['Genotype', 'Treatment', 'Behavior', 'class']
plt.figure(figsize=(30, 20))
for i, column in enumerate(categorical_columns, 1):
    plt.subplot(len(categorical_columns) // 2 + 1, 2, i)
    sns.countplot(x=data[column], palette='viridis',legend =False)
    plt.title(f'Count Plot of {column}', fontsize = 25)
    plt.xlabel(column, fontsize = 20)
    plt.ylabel('Count',fontsize = 20)
plt.tight_layout
plt.show()
# Feature selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
numerical_columns = [col for col in numerical_columns if col not in ['class']] 
# Separate features and target variable
X = data[numerical_columns]
y = data['class']  # Taking 'class' as a target variable
# Standardize the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=numerical_columns)
**Correlation Analysis**
correlation_matrix = X_scaled.corr()
high_corr_pairs = correlation_matrix.abs().unstack().sort_values(kind="quicksort", ascending=False)
high_corr_pairs = high_corr_pairs[(high_corr_pairs > 0.8) & (high_corr_pairs < 1.0)]
print("Highly correlated features:")
print(high_corr_pairs)
**Mutual Information**
mi_scores = mutual_info_classif(X_scaled, y, discrete_features='auto')
mi_scores_df = pd.DataFrame({'Feature': numerical_columns, 'Mutual Information': mi_scores})
mi_scores_df = mi_scores_df.sort_values(by='Mutual Information', ascending=False)
print("Mutual Information Scores:")
print(mi_scores_df)
**Feature Importance from Random Forest**
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)
rf_importances = rf_model.feature_importances_
rf_importance_df = pd.DataFrame({'Feature': numerical_columns, 'Importance': rf_importances})
rf_importance_df = rf_importance_df.sort_values(by='Importance', ascending=False)
print("Random Forest Feature Importances:")
print(rf_importance_df)
**Feature Importance from Gradient Boosting**
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_scaled, y)
gb_importances = gb_model.feature_importances_
gb_importance_df = pd.DataFrame({'Feature': numerical_columns, 'Importance': gb_importances})
gb_importance_df = gb_importance_df.sort_values(by='Importance', ascending=False)
print("Gradient Boosting Feature Importances:")
print(gb_importance_df)
**Feature Importance using Select KBest with F-Test**
best_features = SelectKBest(score_func=f_classif, k='all')
fit = best_features.fit(X_scaled, y)
feature_scores = pd.DataFrame({'Feature': numerical_columns, 'Score': fit.scores_})
feature_scores = (feature_scores.sort_values(by='Score', ascending=False))
print("F-Test Feature Scores:")
print(feature_scores)
**Combining Results**
combined_importance = pd.merge(rf_importance_df, gb_importance_df, on='Feature', how='outer', suffixes=('_RF', '_GB'))
combined_importance['Avg_Importance'] = combined_importance[['Importance_RF', 'Importance_GB']].mean(axis=1)
combined_importance = (combined_importance.sort_values(by='Avg_Importance', ascending=False))
print("Combined Feature Importances:")
print(combined_importance)
**Selecting common columns with the highest feature importances and F-Test scores**
fi_threshold = combined_importance['Avg_Importance'].median()
selected_fi_features = combined_importance[combined_importance['Avg_Importance'] >= fi_threshold]['Feature'].tolist()
fs_threshold = feature_scores['Score'].median()
selected_fs_features = feature_scores[feature_scores['Score'] >= fs_threshold]['Feature'].tolist()
common_selected_features = list(set(selected_fi_features) & set(selected_fs_features))
print("Selected Features based on combined importances and F-Test scores:")
print(common_selected_features)
len((common_selected_features))
# Model Training
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb
**Spliting the dataset into training and testing sets**
X_selected = X_scaled[common_selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
**Model training using Random Forest Classifier**
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=rf_param_grid, cv=5, n_jobs=-1, verbose=2)
rf_grid_search.fit(X_train, y_train)
best_rf = rf_grid_search.best_estimator_
rf_pred = best_rf.predict(X_test)
**Model training using SVM**
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}
svm_grid_search = GridSearchCV(estimator=SVC(random_state=42), param_grid=svm_param_grid, cv=5, n_jobs=-1, verbose=2)
svm_grid_search.fit(X_train, y_train)
best_svm = svm_grid_search.best_estimator_
svm_pred = best_svm.predict(X_test)

**Model training using XGBoost**
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0]
}
xgb_grid_search = GridSearchCV(estimator=xgb.XGBClassifier(random_state=42), param_grid=xgb_param_grid, cv=5, n_jobs=-1, verbose=2)
xgb_grid_search.fit(X_train, y_train)
best_xgb = xgb_grid_search.best_estimator_
xgb_pred = best_xgb.predict(X_test)
# Model Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
**Function for evaluating models & plotting Confusion matrix**
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted')}")
    print(f"Recall: {recall_score(y_test, y_pred, average='weighted')}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()
**Evaluating Random Forest**
print("Random Forest Model Evaluation")
evaluate_model(best_rf, X_test, y_test, "Random Forest Classifier")
**Evaluating SVM**
print("SVM Model Evaluation")
evaluate_model(best_svm, X_test, y_test, "SVM")
**Evaluating XGBoost**
print("XGBoost Model Evaluation")
evaluate_model(best_xgb, X_test, y_test, "XGBoost")
# Interpretation
**Summarizing Model Performance**
def collect_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1 Score': f1_score(y_test, y_pred, average='weighted')
    }
    return metrics
# Collect metrics for each model
rf_metrics = collect_metrics(best_rf, X_test, y_test)
svm_metrics = collect_metrics(best_svm, X_test, y_test)
xgb_metrics = collect_metrics(best_xgb, X_test, y_test)

metrics_df = pd.DataFrame({
    'Model': ['Random Forest', 'SVM', 'XGBoost'],
    'Accuracy': [rf_metrics['Accuracy'], svm_metrics['Accuracy'], xgb_metrics['Accuracy']],
    'Precision': [rf_metrics['Precision'], svm_metrics['Precision'], xgb_metrics['Precision']],
    'Recall': [rf_metrics['Recall'], svm_metrics['Recall'], xgb_metrics['Recall']],
    'F1 Score': [rf_metrics['F1 Score'], svm_metrics['F1 Score'], xgb_metrics['F1 Score']]
})
**Plot bar graphs for each metric**
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
plt.figure(figsize=(12, 8))
for metric in metrics:
    plt.subplot(2, 2, metrics.index(metric) + 1)
    sns.barplot(x='Model', y=metric, data=metrics_df)
    plt.title(f'{metric} Comparison')
    plt.ylim(0, 1) 
plt.tight_layout()
plt.show()
**Analyze Feature Importance**
# Function for plotting feature importance bar graph
def plot_feature_importance(model, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_test.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'{model_name} Feature Importances')
        plt.show()
    else:
        print(f"{model_name} does not support feature importances.")
**Random forest feature importance**
plot_feature_importance(best_rf, "Random Forest")
**XGBoost feature importance**
plot_feature_importance(best_xgb, "XGBoost")
**SVM feature importance**
plot_feature_importance(best_xgb, "SVM")
**Hyperparameter Tuning Results**

def print_best_params(grid_search, model_name):
    print(f"\nBest Hyperparameters for {model_name}:")
    print(grid_search.best_params_)

print_best_params(rf_grid_search, "Random Forest")
print_best_params(svm_grid_search, "SVM")
print_best_params(xgb_grid_search, "XGBoost")

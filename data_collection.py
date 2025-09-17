from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

df = pd.read_parquet("data.parquet")

# Summary stats of the dataset
print("Shape:", df.shape) 
print("\nColumns and dtypes:\n", df.dtypes)
print("\nInfo:")
print(df.info())
print("\nHead:\n", df.head())

# Find the most important features for determining trader_label
label_encoder = LabelEncoder()
df['trader_label_encoded'] = label_encoder.fit_transform(df['trader_label'])

numeric_cols = df.select_dtypes(include='number').columns.drop('trader_label_encoded', errors='ignore')
X = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
y = df['trader_label_encoded']


correlations = X.corrwith(y).abs().sort_values(ascending=False)
mi = mutual_info_classif(X, y, discrete_features=False, random_state=1)
mi_series = pd.Series(mi, index=numeric_cols).sort_values(ascending=False)

# Random Forest Feature Importance
random_forest = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)
random_forest.fit(X, y)
rf_importances = pd.Series(random_forest.feature_importances_, index=numeric_cols).sort_values(ascending=False)

feature_importance = pd.DataFrame({
    'Correlation': correlations,
    'Mutual Information': mi_series,
    'Random Forest Importance': random_forest.feature_importances_
})
feature_scores = feature_importance
feature_scores['Mean_Score'] = feature_scores.mean(axis=1)
top_features = feature_scores.sort_values('Mean_Score', ascending=False).head(10)

print('Top 10 influential features for trader_label:')
print(top_features)


top_feature_names = top_features.index.tolist()
SAMPLE_SIZE = 10000
plot_df = df.sample(SAMPLE_SIZE, random_state=1) if len(df) > SAMPLE_SIZE else df

os.makedirs('feature_importance_table', exist_ok=True)
top_features.to_csv('feature_importance_table/top_features.csv', index=True)

top_features = [
    'trader_ppv', 'trader_pnl', 'mean_delta', 'largest_tags_topic_share', 'std_delta', 'mean_tx_value', 'trader_volume', 'volume_per_day',
    'price_levels_per_transaction', 'std_tx_value'
]

print("\n--- Random Forest Classification on Top Features ---")
# Use only the top features for modeling
X_top = X[top_feature_names]
y = df['trader_label_encoded']

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42, stratify=y)

# Train RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix (Top Features)')
plt.tight_layout()
plt.savefig('results/confusion_matrix_top_features.png')
plt.close()

# --- Overfitting check: train/test/cv accuracy ---
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(clf, X_top, y, cv=5, n_jobs=-1)
print(f"Cross-validation accuracy (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# --- Compare means and sums of top features by trader_label ---
print("\n--- Mean and Sum of Top Features by trader_label ---")
grouped_stats = df.groupby('trader_label')[top_feature_names].agg(['mean'])
print(grouped_stats)

# Optionally, save to CSV for further analysis
grouped_stats.to_csv('results/top_features_grouped_stats.csv')

os.makedirs('results/trend_graphs', exist_ok=True)

key_features = ['trader_pnl', 'trader_ppv', 'mean_tx_value', 'std_tx_value']
means = df.groupby('trader_label')[key_features].mean().reset_index()

for feature in key_features:
    plt.figure(figsize=(7, 5))
    sns.barplot(x='trader_label', y=feature, data=means, order=sorted(df['trader_label'].unique()))
    plt.title(f'Mean {feature} by trader_label')
    plt.ylabel(f'Mean {feature}')
    plt.xlabel('trader_label')
    plt.tight_layout()
    plt.savefig(f'results/trend_graphs/mean_{feature}_by_trader_label.png')
    plt.close()

# --- Density (KDE) plots for key features by trader_label ---
SAMPLE_SIZE = 3000
sample_df = df.sample(SAMPLE_SIZE, random_state=42) if len(df) > SAMPLE_SIZE else df

for feature in key_features:
    plt.figure(figsize=(8, 5))
    for label in sorted(sample_df['trader_label'].unique()):
        subset = sample_df[sample_df['trader_label'] == label]
        sns.kdeplot(subset[feature], label=label, fill=True, common_norm=False, alpha=0.4)
    plt.title(f'Density Plot of {feature} by trader_label')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend(title='trader_label')
    if feature == 'trader_pnl':
        plt.xlim(-600, 600)
    elif feature == 'mean_tx_value':
        plt.xlim(-300, 300)
    elif feature == 'std_tx_value':
        plt.xlim(-300, 300)
    elif feature == 'trader_ppv':
        plt.xlim(-0.75, 0.75)
    plt.tight_layout()
    plt.savefig(f'results/trend_graphs/density_{feature}_by_trader_label.png')
    plt.close()

# --- Save mean and quantiles of all features by trader_label to a single CSV ---
exclude_cols = ['trader_label', 'trader_label_encoded']
feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype != 'O']
grouped = df.groupby('trader_label')[feature_cols]
means = grouped.mean()
quantiles = grouped.quantile([0.25, 0.5, 0.75]).unstack(level=-1)
quantiles.columns = [f'{stat}_{feature}' for feature, stat in quantiles.columns]
combined = pd.concat([means, quantiles], axis=1)
combined.to_csv('results/all_features_mean_and_quantiles_by_trader_label.csv')
print('\nSaved mean, 25th, 50th (median), and 75th percentiles of all features by trader_label to results/all_features_mean_and_quantiles_by_trader_label.csv')
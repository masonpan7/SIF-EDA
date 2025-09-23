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
from sklearn.model_selection import cross_val_score

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

# Correlation and Mutual Information
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

# Random Forest Classification on Top Features
print("\nRandom Forest Classification on Top Features")

X_top = X[top_feature_names]
y = df['trader_label_encoded']
X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42, stratify=y)

# Train RandomForest and predict
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

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

# Check for Overfitting on Random Forest Model
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
print(f"\nTrain accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
cv_scores = cross_val_score(clf, X_top, y, cv=5, n_jobs=-1)
print(f"Cross-validation accuracy (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Gathers mean and sum of top 10 important features
print("\nMean and Sum of Top Features")
grouped_stats = df.groupby('trader_label')[top_feature_names].agg(['mean'])
print(grouped_stats)
grouped_stats.to_csv('results/top_features_grouped_stats.csv')

os.makedirs('results/trend_graphs', exist_ok=True)
key_features = ['trader_pnl', 'trader_ppv', 'mean_tx_value', 'std_tx_value']
means = df.groupby('trader_label')[key_features].mean().reset_index()

# Bar Plots for mean values of trader_pnl, trader_ppv, mean_tx_value, std_tx_value
for feature in key_features:
    plt.figure(figsize=(7, 5))
    sns.barplot(x='trader_label', y=feature, data=means, order=sorted(df['trader_label'].unique()))
    plt.title(f'Mean {feature} by trader_label')
    plt.ylabel(f'Mean {feature}')
    plt.xlabel('trader_label')
    plt.tight_layout()
    plt.savefig(f'results/trend_graphs/mean_{feature}_by_trader_label.png')
    plt.close()

# Density Plots for trader_pnl, trader_ppv, mean_tx_value, std_tx_value
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

# Analyze largest_tags_topic_share
print("\nTopic Share Analysis:")
topic_stats = df.groupby('trader_label')['largest_tags_topic_share'].agg(['mean', 'median'])
print(topic_stats)

# Price levels per transaction
print("\nPrice Levels Per Transaction Analysis:")
price_stats = df.groupby('trader_label')['price_levels_per_transaction'].agg(['mean', 'std'])
print(price_stats)

# Volume per day consistency
print("\nVolume Per Day Analysis:")
volume_stats = df.groupby('trader_label')['volume_per_day'].agg(['mean', 'median', 'std'])
print(volume_stats)

# Calculate coefficient of variation for consistency measure
df['volume_consistency'] = df.groupby('trader_label')['volume_per_day'].transform(
    lambda x: x.std() / x.mean() if x.mean() != 0 else 0
)

consistency_by_category = df.groupby('trader_label')['volume_consistency'].mean()

# PnL to Volume ratio
df['pnl_volume_ratio'] = df['trader_pnl'] / (df['trader_volume'] + 1)  

# Transaction value consistency
df['tx_value_consistency'] = df['mean_tx_value'] / (df['std_tx_value'] + 1)

# Delta efficiency 
df['delta_efficiency'] = df['mean_delta'] / (df['std_delta'] + 1)

ratio_features = ['pnl_volume_ratio', 'tx_value_consistency', 'delta_efficiency']
ratio_stats = df.groupby('trader_label')[ratio_features].mean()
print(ratio_stats)

# Trading efficiency 
print("\nTrading Efficiency:")

# PnL per price level
df['pnl_per_price_level'] = df['trader_pnl'] / (df['price_levels_per_transaction'] + 1)

# Volume efficiency
df['daily_volume_efficiency'] = df['volume_per_day'] / (df['trader_volume'] + 1)

efficiency_features = ['pnl_per_price_level', 'daily_volume_efficiency']
efficiency_stats = df.groupby('trader_label')[efficiency_features].mean()
print(efficiency_stats)

engineered_features = ratio_features + efficiency_features

# Correlation analysis of engineered features with trader_label
print("\nCorrelation Analysis of Engineered Features:")

le = LabelEncoder()
df['trader_label_encoded'] = le.fit_transform(df['trader_label'])

new_features = ratio_features + efficiency_features
correlations = df[new_features + ['trader_label_encoded']].corr()['trader_label_encoded'].drop('trader_label_encoded')
correlations = correlations.sort_values(ascending=False)
print(correlations)


os.makedirs('key_analysis', exist_ok=True)
df['pnl_volume_ratio'] = df['trader_pnl'] / (df['trader_volume'] + 1)
df['pnl_per_price_level'] = df['trader_pnl'] / (df['price_levels_per_transaction'] + 0.001)

# Calculate volume consistency
df['volume_consistency'] = df.groupby('trader_label')['volume_per_day'].transform(
    lambda x: x.std() / x.mean() if x.mean() != 0 else 0
)

# Bar Chart for PnL per Price Level
plt.figure(figsize=(10, 6))
pnl_per_level_means = df.groupby('trader_label')['pnl_per_price_level'].median()
bars = plt.bar(range(len(pnl_per_level_means)), pnl_per_level_means.values, 
               color=['#C62828', '#F57C00', '#388E3C', '#1B5E20'])
plt.xticks(range(len(pnl_per_level_means)), ['awful', 'bad', 'good', 'sharp'])
plt.title('PnL per Price Level by Category')
plt.ylabel('PnL per Price Level')
plt.xlabel('Trader Category')
plt.tight_layout()
plt.savefig('key_analysis/pnl_per_price_level.png', dpi=300)
plt.close()

# Box ploy for Topic Share Distribution
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='trader_label', y='largest_tags_topic_share', 
            order=['awful', 'bad', 'good', 'sharp'])
plt.title('Topic Share Distribution by Category')
plt.ylabel('Largest Tags Topic Share')
plt.xlabel('Trader Category')
plt.tight_layout()
plt.savefig('key_analysis/topic_share_distribution.png', dpi=300)
plt.close()

# 3. Bar Chart fo PnL per Volume Unit
plt.figure(figsize=(10, 6))
pnl_volume_means = df.groupby('trader_label')['pnl_volume_ratio'].mean()
bars = plt.bar(range(len(pnl_volume_means)), pnl_volume_means.values,
               color=['#C62828', '#F57C00', '#388E3C', '#1B5E20'])
plt.xticks(range(len(pnl_volume_means)), ['awful', 'bad', 'good', 'sharp'])
plt.title('PnL per Volume Unit by Category')
plt.ylabel('PnL / Volume Ratio')
plt.xlabel('Trader Category')
plt.tight_layout()
plt.savefig('key_analysis/pnl_volume_ratio.png', dpi=300)
plt.close()

# Scatter ploy for Volume Consistency vs Performance
plt.figure(figsize=(10, 6))
sample_size = 5000
sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)

colors = {'awful': 'red', 'bad': 'orange', 'good': 'lightgreen', 'sharp': 'green'}
for category in ['awful', 'bad', 'good', 'sharp']:
    subset = sample_df[sample_df['trader_label'] == category]
    plt.scatter(subset['volume_consistency'], subset['trader_pnl'], 
                c=colors[category], alpha=0.6, label=category, s=20)

plt.xlabel('Volume Consistency')
plt.ylabel('Trader PnL')
plt.title('Volume Consistency vs Performance')
plt.legend(title='Trader Category')
plt.xlim(0, 50)
plt.ylim(-2000, 2000)  
plt.tight_layout()
plt.savefig('key_analysis/volume_consistency_vs_performance.png', dpi=300)
plt.close()

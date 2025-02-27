# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn import metrics

df = pd.read_csv('Adult_income_dataset.csv')
print(df.head())

## Data Information

print(df.shape)

print(df.describe(include='all'))

print(df.infer_objects().dtypes)

print(df.duplicated().sum())

print(df.isna().sum())


## Data Pre-Processing

df = df.replace('?', np.nan).dropna()

print(df.drop_duplicates().shape)

## Numerical Variable distributions

numeric_var = ['age','Final_Weight_of_Income','education.num','capital.gain','capital.loss','hours.per.week']

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20,10))
for ax, col in zip(axs.flatten(), numeric_var):
    norm_df = np.log(df[col] + 1)
    sns.histplot(norm_df, kde=True, ax=ax)
    ax.set_title(f'{col.capitalize()} Distribution')
    ax.set_ylabel('Count')
    plt.show()

## Distribution by Income

columns = df.drop(columns=numeric_var).columns
fig, axs = plt.subplots(2, 3, figsize=(34,18))
for ax, col in zip(axs.flatten(), columns):
    sns.countplot(x=col, data=df, hue='income', ax=ax)
    ax.set_title(f'{col.capitalize()} vs Income')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    plt.show()

## Distribution by Race

columns = df.drop(columns=numeric_var).columns
fig, axs = plt.subplots(2, 3, figsize=(32, 18))

for ax, col in zip(axs.flatten(), columns):
    crosstab = pd.crosstab(df[col], df['race'])
    crosstab.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title(f'{col.capitalize()} vs Race')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=30)
    plt.show()

## Distribution by Gender

columns = df.drop(columns=numeric_var).columns
fig, axs = plt.subplots(2, 3, figsize=(32, 18))

for ax, col in zip(axs.flatten(), columns):
    crosstab = pd.crosstab(df[col], df['sex'])
    crosstab.plot(kind='bar', stacked=True, ax=ax, colormap='Set1')
    ax.set_title(f'{col.capitalize()} vs Gender')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=30)
    plt.show()

## Numerical Values over Age

plt.figure(figsize=(15,8))
for col in numeric_var:
    if col != 'age':
        sns.lineplot(x = df['age'], y=np.log(df[col] + 1), label = col.capitalize(), errorbar=None)
        plt.ylabel('')
        plt.show()
    else:
        continue

## Boxplot

plt.figure(figsize=(32,15))
sns.boxplot(x='workclass', y='age',  hue='workclass', data=df)
plt.title('Age vs Workclass')

plt.figure(figsize=(32,15))
sns.boxplot(x='race', y='age', hue='race', data=df)
plt.title('Age vs Race')
plt.show()

## Correlation Matrix

corr = df[numeric_var].corr()
plt.figure(figsize=(12,5))
sns.heatmap(corr, vmin=0, vmax=1, cmap='viridis',
            annot=True, fmt='.2f')
plt.show()


## Data Processing

encoder = LabelEncoder()
column_to_encode = df.drop(columns=numeric_var).columns
for col in column_to_encode:
    df[col] = encoder.fit_transform(df[col])


## Principal Component Analysis (PCA)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

pca = PCA(random_state=42)

pca_data = pd.DataFrame(pca.fit_transform(scaled_data), columns=[f'PC{i+1}' for i in range(len(pca.explained_variance_))])

explained_var = pca.explained_variance_ratio_

print(f'These is the amount of variance explained by each PCA: \n{explained_var}')

print('\nThe Head of the PCA Data')
pca_data.head()

per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

plt.figure(figsize=(12, 8))
plt.bar(range(1, len(per_var) + 1), height=per_var)

plt.ylabel('Percentage of Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('Scree Plot')
plt.xticks(ticks=range(1, len(labels) + 1), labels=labels, rotation=90)
plt.show()


## Model Train Evaluation


X_train, X_val, y_train, y_val = train_test_split(pca_data, df['income'], test_size=0.2, random_state=42)

logreg = LogisticRegression(random_state=42)
logreg.fit(X_train, y_train)

y_test_pred = logreg.predict(X_val)
y_train_pred = logreg.predict(X_train)

print(f'Train score is: {metrics.accuracy_score(y_train, y_train_pred)} and the Test score is: {metrics.accuracy_score(y_val, y_test_pred)}')

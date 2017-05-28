import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Visualization
import seaborn as sns # For easier statistical plotting
sns.set_style("whitegrid")
from sklearn.model_selection import KFold, cross_val_score, train_test_split # Model evaluation
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler # Preprocessing
from sklearn.linear_model import Lasso, Ridge, ElasticNet, RANSACRegressor # Linear models
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor  # Ensemble methods
from xgboost import XGBRegressor, plot_importance # XGBoost
from sklearn.svm import SVR  # Support Vector Regression
from sklearn.tree import DecisionTreeRegressor # Decision Tree Regression
from sklearn.pipeline import Pipeline # Streaming pipelines
from sklearn.decomposition import KernelPCA, PCA # Dimensionality reduction
from sklearn.feature_selection import SelectFromModel # Dimensionality reduction
from sklearn.model_selection import learning_curve, validation_curve, GridSearchCV # Model evaluation
from sklearn.base import clone # Clone estimator
from sklearn.metrics import mean_squared_error as MSE

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# Get the correlation heatmap
# If may be a hard to analyse
pd.set_option('precision', 2)
plt.figure(figsize=(10, 6))
sns.heatmap(df_train.drop(["SalePrice","Id"],axis=1).corr(), square=True)
plt.xticks(rotation=90)
plt.yticks(rotation=360)
plt.suptitle("Pearson Correlation Heatmap")
plt.show()

# A more straightforward way to do that
corr_with_Saleprice = df_train.drop(["SalePrice", "Id"], axis=1).corr()['SalePrice'].sort_values(ascending=False)
corr_with_Saleprice.drop(["SalePrice"], axis=1, inplace=True)
plt.figure(figsize=(10, 6))
corr_with_Saleprice.plot.bar()
plot.show()
del corr_with_Saleprice

# corr_with_Saleprice = df_train.drop(['Id'], axis=1).corr()['SalePrice'].sort_values(ascending=False)
# plt.figure(figsize=(14, 7))
# corr_with_Saleprice.drop(['SalePrice']).plot.bar()
# plt.show()
# del corr_with_Saleprice

# To analyse the skewness of each features
skews = df_train.skew()
for col in df_train.drop(['Id', 'SalePrice'], axis=1).columns.tolist():
    if df_train[col].dtype == 'float64' or df_train[col].dtype == 'int64':
        sns.distplot(df_train[col].dropna(), kde=True, label="skew= " + str(round(skews[col], 2)))
        plt.legend()
        plt.show()
del skews

# To find the skewness of the SalePrice
skewness = df_train['SalePrice'].skew()
sns.distplot(df_train['SalePrice'], kde=True, label="Skew= " + str(round(skewness, 2)))
plt.legend()
plt.show()

# To make it be more like guassian.
skewness = np.log(df_train['SalePrice'].skew())
np.log(df_train['SalePrice']).plot.hist(edgecolor='white', bins=60, label=str(round(skewness, 2)))
# sns.distplot(np.log(df_train['SalePrice']), kde=True)
plt.legend()
plt.show()

# Size of each features which are as type of object.
for col in df_train.columns.tolist():
    if df_train[col].dtype == 'object':
        sns.countplot(col, data=df_train)
        plt.xticks(rotation=55)
        plt.show()


label_nas = []
for col in df_train.columns.tolist():
    if np.sum(df_train[col].isnull()) != 0:
        label_nas.append(col)
    else:
        label_nas.append("")

plt.figure(figsize=(12, 7))
sns.heatmap(df_train.isnull(), yticklabels=False, xticklabels=label_nas, cmap='viridis')
plt.xticks(rotation=90)
plt.show()

# To fill NaN.

null_values_per_col = np.sum(df_train.drop(["Id", "SalePrice"], axis=1).isnull(), axis=0)
max_na = int(2 * df_train.shape[0] / 3.0)
cols_to_remove = []
cols_to_impute = []

for col in df_train.drop(["Id", "SalePrice"], axis=1).columns.tolist():
    if null_values_per_col[col] > max_na:
        cols_to_remove.append(col)
        df_train.drop(col, axis=1, inplace=True)
        df_test.drop(col, axis=1, inplace=True)
    elif null_values_per_col[col] > 0:
        cols_to_impute.append(col)

imputation_val_for_na_cols = dict()

for col in cols_to_impute:
    if df_train[col].dtype == 'int64' or df_train[col].dtype == 'float64':
        imputation_val_for_na_cols[col] = df_train[col].median()
    else:
        imputation_val_for_na_cols[col] = df_train[col].max()

for col, val in imputation_val_for_na_cols.items():
    df_train[col].fillna(value=val, inplace=True)
    df_test[col].fillna(value=val, inplace=True)

null_values_per_col_for_test = np.sum(df_test.drop(["Id"], axis=1).isnull(), axis=0)
for col in df_test.drop(["Id"], axis=1).columns.tolist():
    if null_values_per_col_for_test[col] > 0:
        if df_test[col].dtype == 'float64' or df_test[col].dtype == 'int64':
            df_test[col].fillna(df_train[col].median(), inplace=True)
        else:
            df_test[col].fillna(df_train[col].max(), inplace=True)

# Data Munging
X, y = df_train.drop(['Id', 'SalePrice'], axis=1), np.log(df_train['SalePrice'])
X_test = df_test.drop(['Id'], axis=1)

categorical_features = []
is_categorical = X.dtypes == 'object'

for col in X.columns.tolist():
    if is_categorical[col]:
        categorical_features.append(col)

# Important way to process!!!!!
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])
    ohe = OneHotEncoder(sparse=False)
    columns = [col + str("_") + str(class_) for class_ in le.classes_]
    X_dummies = pd.DataFrame(ohe.fit_transform(X[col].values.reshape(-1, 1))[:, 1:],
                             columns=columns[1:])
    X_test_dummies = pd.DataFrame(ohe.transform(X_test[col].values.reshape(-1, 1))[:, 1:],
                                  columns=columns[1:])
    X.drop(col, axis=1)
    X_test.drop(col, axis=1)
    X = pd.concat([X, X_dummies], axis=1)
    X_test = pd.concat([X_test, X_test_dummies], axis = 1)
    
    # This is an easy way to get dummies
    # but maybe the value range of train set and test set are different.

    # dummies_train = pd.get_dummies(X[col], prefix=col)
    # dummies_test = pd.get_dummies(X_test[col], prefix=col)
    # X = pd.DataFrame(pd.concat([X, dummies_train], axis=1))
    # X_test = pd.DataFrame(pd.concat([X_test, dummies_test], axis=1))
    # X.drop([col], axis=1, inplace=True)
    # X_test.drop([col], axis=1, inplace=True)


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
del X, df_train

# Select Feature.
thresh = 5 * 10**(-3)
model = XGBRegressor()
model.fit(X_train, y_train)
selection = SelectFromModel(model, threshold=thresh, prefit=True)
select_X_train = selection.transform(X_train)
select_X_val = selection.transform(X_val)
select_X_test = selection.transform(X_test)

pipelines = []

seed = 7

pipelines.append(
    ("Scaled_Ridge",
     Pipeline([
         ("Scaler", StandardScaler()),
         ("Ridge", Ridge(random_state=seed))
     ]))
)

pipelines.append(
    ("Scaled_Lasso",
     Pipeline([
         ("Scaler", StandardScaler()),
         ("Lasso", Lasso(random_state=seed))
     ]))
)

pipelines.append(
    ("Scaled_SVR",
     Pipeline([
         ("Scaler", StandardScaler()),
         ("SVR", SVR())
     ])
     )
)

pipelines.append(
    ("Scaled_RF",
     Pipeline([
         ("Scaler", StandardScaler()),
         ("RF", RandomForestRegressor(random_state=seed))
     ])
     )
)

pipelines.append(
    ("Scaled_ET",
     Pipeline([
         ("Scaler", StandardScaler()),
         ("ET", ExtraTreesRegressor(random_state=seed))
     ])
     )
)
pipelines.append(
    ("Scaled_BR",
     Pipeline([
         ("Scaler", StandardScaler()),
         ("BR", BaggingRegressor(random_state=seed))
     ])
     )
)

pipelines.append(
    ("Scaled_XGB",
     Pipeline([
         ("Scaler", StandardScaler()),
         ("XGB", XGBRegressor(random_state=seed))
     ])
     )
)

scoring = 'r2'
n_folds = 10
results, names = [], []

for name, model in pipelines:
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, select_X_train, y_train, cv=kfold,
                                 scoring=scoring, n_jobs=-1)
    names.append(name)
    results.append(cv_results)
    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

results = pd.DataFrame(np.array(results).T, columns=names)
sns.boxplot(results)
plt.show()

Scaled_XGB = pipelines[-1][1]

# param_grid_rf = [{
#     'RF__n_estimators': [2500],
#     'RF__max_depth': np.arange(5, 15, 2),
#     'RF__max_features': ['sqrt'],
#     'RF__min_samples_leaf': [0.04]
# }]


# grid_rf = GridSearchCV(estimator=Scaled_RF,
#                        param_grid=param_grid_rf,
#                        scoring='neg_mean_squared_error',
#                        cv=KFold(n_splits=3, random_state=seed, shuffle=True),
#                        verbose=1)
# Fit grid
# grid_rf.fit(select_X_train, y_train)

# Best score and best parameters
# print('-------Best score----------')
# print(grid_rf.best_score_)
# print('-------Best params----------')
# print(grid_rf.best_params_)


def plot_learning_curve(train_sizes, train_scores, test_scores, title, alpha=0.1):
    train_scores = -train_scores
    test_scores = -test_scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(train_sizes,train_mean + train_std,
                    train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(train_sizes, test_mean, label='test score', color='red',marker='o')
    plt.fill_between(train_sizes,test_mean + test_std, test_mean - test_std , color='red', alpha=alpha)
    plt.title(title)
    plt.xlabel('Number of training points')
    plt.ylabel(r'Mean Squared Error')
    plt.grid(ls='--')
    plt.legend(loc='best')
    plt.show() 

# Plot the learning curve
plt.figure(figsize=(9, 6))
train_sizes, train_scores, test_scores = learning_curve(
    Scaled_XGB, X=select_X_train, y=y_train,
    cv=3, scoring='neg_mean_squared_error')

plot_learning_curve(train_sizes, train_scores, test_scores, title='Learning curve for RF Regressor')

best_model = clone(Scaled_XGB)
best_model.fit(select_X_train, y_train)
# y_pred_train = best_model.predict(select_X_train)


y_pred_test = best_model.predict(select_X_test)
y_pred_test = np.exp(y_pred_test)

df_submission = pd.DataFrame({ "Id": df_test["Id"].values, "SalePrice": y_pred_test })

df_submission.to_csv("housePrice.csv", index= False)

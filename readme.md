# House Price Prediction

这是一个Regression问题，预测房价。本model主要运用了以下重要的feature procession方法。

## fill NaN
简单的处理方法就是，对数值型feature，直接填充中间值；对object型feature，填充最大值。
这是一种比较不容易出现错误的方法，然而更准确的方法是，根据heatmap判断该feature与哪些feature相关性更高，从而建立分类模型。
对于空缺值太大的feature则直接删除该feature，这是可以理解的，因为如果该feature空缺值太大，也就意味着该feature的值难以被预测，就算可以预测也可能造成noise。最好的方法就是不考虑这个feature。

```
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
```


## 离散化object数据

一种简单的方法是直接调用get_dummies。但在实践中出现了问题。
关键就在于train set和test set在该feature上的取值范围很可能是不一样的。

```
    dummies_train = pd.get_dummies(X[col], prefix=col)
    dummies_test = pd.get_dummies(X_test[col], prefix=col)
    X = pd.DataFrame(pd.concat([X, dummies_train], axis=1))
    X_test = pd.DataFrame(pd.concat([X_test, dummies_test], axis=1))
    X.drop([col], axis=1, inplace=True)
    X_test.drop([col], axis=1, inplace=True)
```

为了解决这个问题，可以调用OneHotEncoder方法。

```
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
```

## Feature Selection
这里可以采用PCA，但PCA比较适合于线性关系的减维。这里采用的是XGBoost，取出那些重要性小于0.5%的feature。

```
thresh = 5 * 10**(-3)
model = XGBRegressor()
model.fit(X_train, y_train)
selection = SelectFromModel(model, threshold=thresh, prefit=True)
select_X_train = selection.transform(X_train)
select_X_val = selection.transform(X_val)
select_X_test = selection.transform(X_test)
```

## 选择模型

选择一系列的模型分别对数据集进行测试。

```
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
```

## 参数调优
对最优模型进行参数调优。

```
param_grid_rf = [{
    'RF__n_estimators': [2500],
    'RF__max_depth': np.arange(5, 15, 2),
    'RF__max_features': ['sqrt'],
    'RF__min_samples_leaf': [0.04]
}]


grid_rf = GridSearchCV(estimator=Scaled_RF,
                       param_grid=param_grid_rf,
                       scoring='neg_mean_squared_error',
                       cv=KFold(n_splits=3, random_state=seed, shuffle=True),
                       verbose=1)
Fit grid
grid_rf.fit(select_X_train, y_train)

Best score and best parameters
print('-------Best score----------')
print(grid_rf.best_score_)
print('-------Best params----------')
print(grid_rf.best_params_)

```

## 绘制learning Curve
See The code。

```
def plot_learning_curve(train_sizes, train_scores, test_scores, title, alpha=0.1):
    train_scores = -train_scores
    test_scores = -test_scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean + train_std,
                     train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(train_sizes, test_mean, label='test score', color='red', marker='o')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
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
```

# Reference
[Regression to Predict House Prices](https://www.kaggle.com/eliekawerk/regression-to-predict-house-prices/notebook)

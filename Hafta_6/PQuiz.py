from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

# load_boston is deprecated, using California housing instead
data = fetch_california_housing()
X, y = data.data, data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

ridge_params = {
    'alpha': [0.1, 1.0, 10.0],
    'solver': ['auto', 'svd', 'cholesky'],
    'fit_intercept': [True, False]
}

lasso_params = {
    'alpha': [0.01, 0.1, 1.0],
    'max_iter': [5000, 10000, 20000],
    'fit_intercept': [True, False]
}

ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)

lasso_grid = GridSearchCV(Lasso(), lasso_params, cv=5, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)

ridge_pred = ridge_grid.predict(X_test)
lasso_pred = lasso_grid.predict(X_test)

ridge_mse = mean_squared_error(y_test, ridge_pred)
lasso_mse = mean_squared_error(y_test, lasso_pred)

print("Ridge en iyi parametreler:", ridge_grid.best_params_)
print("Ridge test MSE:", ridge_mse)

print("\nLasso en iyi parametreler:", lasso_grid.best_params_)
print("Lasso test MSE:", lasso_mse)

if ridge_mse < lasso_mse:
    print("\nRidge modeli daha iyi performans gösteriyor. Ridge MSE daha düşük, model daha stabil olabilir.")
else:
    print("\nLasso modeli daha iyi performans gösteriyor. Lasso MSE daha düşük, bazı gereksiz özellikleri sıfırlayarak daha sade model oluşturuyor.")
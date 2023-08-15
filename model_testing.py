#Jacob Thomas Vespers
#jrbiltmore@icloud.com

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge

# Read the dataset
data = pd.read_csv("/content/placement (1).csv")

# Scatter plot of the data
plt.scatter(data['cgpa'], data['package'])
plt.xlabel('CGPA')
plt.ylabel('Package')
plt.title('Scatter Plot: CGPA vs Package')
plt.show()

# Linear Regression
x = data.iloc[:, 0:1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

lr = LinearRegression()
lr.fit(x_train, y_train)

# Plot linear regression line
plt.scatter(x, y)
plt.plot(x, lr.predict(x), color='red')
plt.xlabel('CGPA')
plt.ylabel('Package')
plt.title('Linear Regression: CGPA vs Package')
plt.show()

# Model evaluation
y_pred = lr.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Linear Regression:")
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print()

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x_train)
poly_lr = LinearRegression()
poly_lr.fit(x_poly, y_train)

# Plot polynomial regression curve
plt.scatter(x, y)
plt.plot(x, poly_lr.predict(poly.transform(x)), color='red')
plt.xlabel('CGPA')
plt.ylabel('Package')
plt.title('Polynomial Regression: CGPA vs Package')
plt.show()

# Model evaluation for polynomial regression
y_poly_pred = poly_lr.predict(poly.transform(x_test))
poly_mse = mean_squared_error(y_test, y_poly_pred)
poly_r2 = r2_score(y_test, y_poly_pred)
print("Polynomial Regression:")
print("Mean Squared Error:", poly_mse)
print("R-squared:", poly_r2)
print()

# Decision Tree Regression
dt_regressor = DecisionTreeRegressor(random_state=2)
dt_regressor.fit(x_train, y_train)

# Plot decision tree regression
x_grid = np.arange(min(x_train.values), max(x_train.values), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y)
plt.plot(x_grid, dt_regressor.predict(x_grid), color='red')
plt.xlabel('CGPA')
plt.ylabel('Package')
plt.title('Decision Tree Regression: CGPA vs Package')
plt.show()

# Model evaluation for decision tree regression
y_dt_pred = dt_regressor.predict(x_test)
dt_mse = mean_squared_error(y_test, y_dt_pred)
dt_r2 = r2_score(y_test, y_dt_pred)
print("Decision Tree Regression:")
print("Mean Squared Error:", dt_mse)
print("R-squared:", dt_r2)
print()

# Random Forest Regression
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=2)
rf_regressor.fit(x_train, y_train)

# Plot random forest regression curve
plt.scatter(x, y)
plt.plot(x_grid, rf_regressor.predict(x_grid), color='red')
plt.xlabel('CGPA')
plt.ylabel('Package')
plt.title('Random Forest Regression: CGPA vs Package')
plt.show()

# Model evaluation for random forest regression
y_rf_pred = rf_regressor.predict(x_test)
rf_mse = mean_squared_error(y_test, y_rf_pred)
rf_r2 = r2_score(y_test, y_rf_pred)
print("Random Forest Regression:")
print("Mean Squared Error:", rf_mse)
print("R-squared:", rf_r2)
print()

# Support Vector Regression
svr_regressor = SVR(kernel='linear')
svr_regressor.fit(x_train, y_train)

# Plot support vector regression curve
plt.scatter(x, y)
plt.plot(x, svr_regressor.predict(x), color='red')
plt.xlabel('CGPA')
plt.ylabel('Package')
plt.title('Support Vector Regression: CGPA vs Package')
plt.show()

# Model evaluation for support vector regression
y_svr_pred = svr_regressor.predict(x_test)
svr_mse = mean_squared_error(y_test, y_svr_pred)
svr_r2 = r2_score(y_test, y_svr_pred)
print("Support Vector Regression:")
print("Mean Squared Error:", svr_mse)
print("R-squared:", svr_r2)
print()

# K-Nearest Neighbors Regression
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(x_train, y_train)

# Plot KNN regression curve
plt.scatter(x, y)
plt.plot(x, knn_regressor.predict(x), color='red')
plt.xlabel('CGPA')
plt.ylabel('Package')
plt.title('K-Nearest Neighbors Regression: CGPA vs Package')
plt.show()

# Model evaluation for KNN regression
y_knn_pred = knn_regressor.predict(x_test)
knn_mse = mean_squared_error(y_test, y_knn_pred)
knn_r2 = r2_score(y_test, y_knn_pred)
print("K-Nearest Neighbors Regression:")
print("Mean Squared Error:", knn_mse)
print("R-squared:", knn_r2)
print()

# Lasso Regression
lasso_regressor = Lasso(alpha=0.01)
lasso_regressor.fit(x_train, y_train)

# Plot Lasso regression line
plt.scatter(x, y)
plt.plot(x, lasso_regressor.predict(x), color='red')
plt.xlabel('CGPA')
plt.ylabel('Package')
plt.title('Lasso Regression: CGPA vs Package')
plt.show()

# Model evaluation for Lasso regression
y_lasso_pred = lasso_regressor.predict(x_test)
lasso_mse = mean_squared_error(y_test, y_lasso_pred)
lasso_r2 = r2_score(y_test, y_lasso_pred)
print("Lasso Regression:")
print("Mean Squared Error:", lasso_mse)
print("R-squared:", lasso_r2)
print()

# Ridge Regression
ridge_regressor = Ridge(alpha=1.0)
ridge_regressor.fit(x_train, y_train)

# Plot Ridge regression line
plt.scatter(x, y)
plt.plot(x, ridge_regressor.predict(x), color='red')
plt.xlabel('CGPA')
plt.ylabel('Package')
plt.title('Ridge Regression: CGPA vs Package')
plt.show()

# Model evaluation for Ridge regression
y_ridge_pred = ridge_regressor.predict(x_test)
ridge_mse = mean_squared_error(y_test, y_ridge_pred)
ridge_r2 = r2_score(y_test, y_ridge_pred)
print("Ridge Regression:")
print("Mean Squared Error:", ridge_mse)
print("R-squared:", ridge_r2)
print()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mglearn
from sklearn.linear_model import Lasso
import numpy as np
# Veri oluştur
X, y = mglearn.datasets.make_wave(n_samples=60)

# Veri setini böl
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Model oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# Sonuçları kontrol et
print("Eğitim seti skoru:", model.score(X_train, y_train))
print("Test seti skoru:", model.score(X_test, y_test))


from sklearn.linear_model import Ridge
ridge = Ridge(alpha=10).fit(X_train, y_train)
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))


lasso = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()

print("Keys of iris_dataset: \n{}" .format(iris_dataset.keys()))


X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))
y_pred =knn.predict(X_new)
print("Predicted target value: {}".format(y_pred))
print("Predicted target name: {}".format(iris_dataset['target_names'][y_pred]))
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

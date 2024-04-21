from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas

df = pandas.read_csv("data/housing.csv", header=None)
df.columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
train, test = train_test_split(df, test_size=0.2)
X_train, X_test = train.drop("MEDV", axis=1), test.drop("MEDV", axis=1)
y_train, y_test = train["MEDV"], test["MEDV"]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(mean_squared_error(y_test, y_pred))
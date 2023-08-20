import numpy as np
import pandas as pd
data=pd.read_csv("IRIS.csv")
data.head(10)
from sklearn.ensemble import RandomForestClassifier
target = data['species']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data,target, test_size = 0.5)
y_train = data["species"]

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
x_train = pd.get_dummies(data[features])
x_test = pd.get_dummies(data[features])

my_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
my_model.fit(x_train, y_train)
Y_predict = my_model.predict(x_test)
output = pd.DataFrame({'sepal_length': data.sepal_length,'sepal_width': data.sepal_width,'petal_length': data.sepal_length ,'petal_width': data.petal_width,'species': Y_predict})
output.to_csv('iris_flower_model_predict.csv', index=False)
print("The file is created")

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

iris_data = load_iris() 
print(iris_data.data[:5])
print(iris_data.target[:5])

x = iris_data.data
Y = iris_data.target.reshape(-1, 1) 

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(Y)

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)


model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax', name='output'))
optimizer = Adam(lr=0.001)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=5, epochs=25)
results = model.evaluate(test_x, test_y)


y_pred = model.predict(test_x)
print(y_pred)
y_pred = (y_pred > 0.5)
print(y_pred)



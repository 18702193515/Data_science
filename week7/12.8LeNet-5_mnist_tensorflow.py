import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

batch_size = 128
epochs = 5

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# 定义LeNet-5模型
tf_model = Sequential()
tf_model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
tf_model.add(MaxPooling2D(pool_size=(2, 2)))
tf_model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
tf_model.add(MaxPooling2D(pool_size=(2, 2)))
tf_model.add(Flatten())
tf_model.add(Dense(120, activation='relu'))
tf_model.add(Dense(84, activation='relu'))
tf_model.add(Dense(10, activation='softmax'))

tf_model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

tf_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

loss, accuracy = tf_model.evaluate(x_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")
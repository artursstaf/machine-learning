from keras import Sequential
from keras.datasets import mnist
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
sample2 = x_train[0]
x_train /= 255
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.astype('float32')
x_test /= 255
x_test = x_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential(
        [
            Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='softmax'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(10, activation='softmax')
        ]
)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy') # sparse ?
#model.fit(x_train, y_train, epochs=50, batch_size=100, verbose=1)
#model.save_weights('mnist_weights2')
model.load_weights('mnist_weights')

result = model.evaluate(x_test, y_test, verbose=1)

sample = x_test[0].reshape(28,28)
plt.imshow(x_test[0].reshape(28, 28))
plt.show()

print(model.predict(np.array([x_test[0]])))
print(result)

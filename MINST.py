from keras import Sequential
from keras.datasets import mnist
from keras.layers import Flatten, Dense
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_train /= 255
x_test = x_test.astype('float32')
x_test /= 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential(
        [
            Flatten(input_shape=(28, 28)),
            Dense(70, activation='softmax'),
            Dense(35, activation='softmax'),
            Dense(10, activation='softmax'),
        ]
)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x_train, y_train, epochs=20000, batch_size=1000, verbose=0)
model.save_weights('mnist_weights')

result = model.evaluate(x_test, y_test, verbose=1)
print(result)

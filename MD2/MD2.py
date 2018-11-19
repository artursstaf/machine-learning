import numpy as np
import os
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Dice:
    model = Sequential([
        Conv2D(64, (5, 5), input_shape=(20, 20, 1), activation='relu'),
        MaxPooling2D(pool_size=(3, 3)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(13, activation='softmax')
    ])

    def __init__(self, load=False):
        if load:
            self.model.load_weights('md2')
        else:
            data, answers = self._load_data([Dice._read_vectors("data/dice_0{}.dat".format(i)) for i in range(9)])
            self.model.compile(optimizer='adam', loss='categorical_crossentropy')
            for i in range(60):
                self.model.fit(data, answers, epochs=1, batch_size=1028, verbose=1)
                self.model.save_weights('md2')

    def benchmark_model(self, filename):
        data, answers = self._load_data([Dice._read_vectors(filename)])
        errors = 0
        for sample_data, sample_answer in zip(data, answers):
            result = np.argmax(self.model.predict(np.array([sample_data])))
            if np.argmax(sample_answer) != result:
                plt.imshow(sample_data.reshape(20, 20))
                plt.show()
                errors += 1
        print(errors)
        return (1 - errors / len(data)) * 100

    @staticmethod
    def _load_data(vectors):
        dice = np.vstack(tuple(vectors))
        return dice[:, 1:].reshape(len(dice), 20, 20, 1) / 255, to_categorical(dice[:, 0])

    @staticmethod
    def _read_vectors(filename):
        return np.fromfile(filename, dtype=np.uint8).reshape(-1, 401)


if __name__ == "__main__":
    dice = Dice(True)
    print(dice.benchmark_model('data/dice_00.dat'))

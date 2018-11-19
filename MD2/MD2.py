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
        """
        :param load: should model weights be loaded from file 'md2'
        """
        if load:
            self.model.load_weights('md2')
        else:
            # load only 0-8 files, last(9) used for benchmarking
            data, answers = self.load_data([Dice.read_vectors("data/dice_0{}.dat".format(i)) for i in range(9)])
            self.model.compile(optimizer='adam', loss='categorical_crossentropy')
            for i in range(60):
                self.model.fit(data, answers, epochs=1, batch_size=1028, verbose=1)
                self.model.save_weights('md2')

    def benchmark_model(self, filename):
        """
        :param filename: file name to load test samples from
        :return: number of prediction errors
        """
        data, answers = self.load_data([Dice.read_vectors(filename)])
        errors = 0
        for sample_data, sample_answer in zip(data, answers):
            result = np.argmax(self.model.predict(np.array([sample_data])))
            if np.argmax(sample_answer) != result:
                errors += 1
        print("{} erros from {} test samples, {}% precision".format(errors, len(data), (1 - errors / len(data)) * 100))
        return errors

    def predict(self, image):
        """
        :param image: 20 by 20 by 1 numpy array
        :return: dice value sum
        """
        plt.imshow(image.reshape(20, 20))
        plt.show()
        return np.argmax(self.model.predict(np.array([image])))

    @staticmethod
    def load_data(vectors):
        """
        :param vectors: x by 401 numpy array
        :return: ready for training images and labels
        """
        dice = np.vstack(tuple(vectors))
        return dice[:, 1:].reshape(len(dice), 20, 20, 1) / 255, to_categorical(dice[:, 0])

    @staticmethod
    def read_vectors(filename):
        """
        :param filename: path to .dat file containing samples
        :return: raw numpy array containing samples
        """
        return np.fromfile(filename, dtype=np.uint8).reshape(-1, 401)


if __name__ == "__main__":
    dice = Dice(True)
    dice.benchmark_model('data/dice_09.dat')
    image = Dice.load_data(Dice.read_vectors('data/dice_09.dat'))[0][0]
    image2 = Dice.load_data(Dice.read_vectors('data/dice_09.dat'))[0][5]
    print(dice.predict(image))
    print(dice.predict(image2))

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
            data, answers = self.load_data([f'data/dice_0{i}.dat' for i in range(9)])
            self.model.compile(optimizer='adam', loss='categorical_crossentropy')
            for i in range(60):
                self.model.fit(data, answers, epochs=1, batch_size=1028, verbose=1)
                self.model.save_weights('md2')

    def benchmark_model(self, filename):
        """
        :param filename: file name to load test samples from
        :return: number of prediction errors
        """
        data, answers = self.load_data(filename)
        errors = 0
        for sample_data, sample_answer in zip(data, answers):
            result = np.argmax(self.model.predict(np.array([sample_data])))
            if np.argmax(sample_answer) != result:
                errors += 1

        print(f'{errors} errors from {len(data)} test samples, {1 - errors / len(data) * 100} % precision')
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
    def load_data(source):
        """
        :param source: file or list of file names containing dice samples
        :return: ready for training (images, labels) in shape - len(samples) by 20 by 20 by 1
        """
        if isinstance(source, (list,)):
            vectors = [Dice._read_vectors(file) for file in source]
        elif isinstance(source, str):
            vectors = Dice._read_vectors(source)
        else:
            raise ValueError('expecting string or list of strings')

        stacked_vector = np.vstack(tuple(vectors))
        return stacked_vector[:, 1:].reshape(len(stacked_vector), 20, 20, 1) / 255, to_categorical(stacked_vector[:, 0])

    @staticmethod
    def _read_vectors(filename):
        return np.fromfile(filename, dtype=np.uint8).reshape(-1, 401)


if __name__ == "__main__":
    dice = Dice(True)
    dice.benchmark_model('data/dice_09.dat')
    image = Dice.load_data('data/dice_09.dat')[0][0]
    image2 = Dice.load_data('data/dice_09.dat')[0][5]
    print(dice.predict(image))
    print(dice.predict(image2))

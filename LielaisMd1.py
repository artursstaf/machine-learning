import numpy as np
import os
from keras.layers import Flatten, Dense
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class RPS:
    def __init__(self, load=False):
        self.model = Sequential(
            [
                Flatten(input_shape=(2, 3)),
                Dense(35, activation='sigmoid'),
                Dense(2, activation='sigmoid')
            ]
        )

        if load:
            self.model.load_weights('rps_model')
        else:
            self.model.compile(optimizer='adam', loss='binary_crossentropy')
            data = np.array([
                # 2 players each have 1 hot encoded [rock, paper, scissors]
                [[0, 0, 1], [0, 0, 1]],
                [[0, 0, 1], [0, 1, 0]],
                [[0, 0, 1], [1, 0, 0]],
                [[0, 1, 0], [0, 0, 1]],
                [[0, 1, 0], [0, 1, 0]],
                [[0, 1, 0], [1, 0, 0]],
                [[1, 0, 0], [0, 0, 1]],
                [[1, 0, 0], [0, 1, 0]],
                [[1, 0, 0], [1, 0, 0]]
            ])
            answers = np.array([
                [1, 1],
                [1, 0],
                [0, 1],
                [0, 1],
                [1, 1],
                [1, 0],
                [1, 0],
                [0, 1],
                [1, 1]
            ])
            self.model.fit(data, answers, epochs=9000, batch_size=9, verbose=1)
            self.model.save_weights('rps_model')

    def play(self, first, second):
        prediction = self.model.predict(np.array([[self._get_one_hot(first), self._get_one_hot(second)]]))[0]
        if prediction[0] > 0.999 and prediction[1] < 0.001:
            print('1.{} vs 2.{} -> first player won'.format(first, second))
        elif prediction[1] > 0.999 and prediction[0] < 0.001:
            print('1.{} vs 2.{} -> second player won'.format(first, second))
        elif prediction[0] > 0.999 and prediction[1] > 0.999:
            print('1.{} vs 2.{} -> draw'.format(first, second))
        else:
            print('Ambiguous result')

    @staticmethod
    def _get_one_hot(choice):
        encoding = {
            'scissors': [0, 0, 1],
            'paper': [0, 1, 0],
            'rock': [1, 0, 0],
        }
        return encoding.get(choice.lower(), 'Invalid choice')


if __name__ == "__main__":
    rps = RPS(load=False)
    rps.play('scissors', 'scissors')
    rps.play('scissors', 'rock')
    rps.play('scissors', 'paper')
    rps.play('rock', 'scissors')
    rps.play('rock', 'rock')
    rps.play('rock', 'paper')
    rps.play('paper', 'scissors')
    rps.play('paper', 'rock')
    rps.play('paper', 'paper')

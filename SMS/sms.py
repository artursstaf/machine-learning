import os
from keras.layers import Flatten, Dense, Embedding, LSTM
from keras.models import Sequential
import pandas as pd
from keras_preprocessing import sequence
from keras_preprocessing.text import Tokenizer
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class SMS:
    def __init__(self, load=False):
        self.model = Sequential(
            [
                Embedding(1000, 32),
                LSTM(64, dropout=0.2, recurrent_dropout=0.2),
                Dense(128, activation='relu'),
                Dense(1, activation='sigmoid'),
            ]
        )

        if load:
            self.model.load_weights('sms_model')
        else:
            self.model.compile(metrics=['acc'], optimizer='adam', loss='binary_crossentropy')
            df = pd.read_csv('14 spam data.csv', delimiter=',', encoding='latin-1')[['v1', 'v2']]
            Y = (df.v1 == 'ham').astype('int').as_matrix()
            X = df.v2
            max_words = 1000
            max_len = 150
            tok = Tokenizer(num_words=max_words)
            tok.fit_on_texts(X)
            sequences = tok.texts_to_sequences(X)
            X = sequence.pad_sequences(sequences, maxlen=max_len)
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

            for i in range(5):
                self.model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)
                self.model.save_weights('sms_model')

            print(self.model.evaluate(X_test, y_test))


if __name__ == "__main__":
    sms = SMS(load=False)

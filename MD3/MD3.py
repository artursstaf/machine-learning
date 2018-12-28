import os

import numpy as np
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.datasets import imdb
from keras.engine.saving import load_model
from keras.layers import Embedding, LSTM, Lambda, SpatialDropout1D
from keras_preprocessing import sequence
from keras_preprocessing.text import text_to_word_sequence

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ImdbNN:
    def __init__(self, load=False):
        (self.word_to_id, self.id_to_word) = self.get_imdb_word_mappings()
        self.max_features = 20000
        self.max_length = 400
        self.skip_top = 20

        model_file = "imdb_model.h5"
        model_predict_file = "imdb_predict_model.h5"

        if load:
            self.model = load_model(model_file)
            self.model_predict = load_model(model_predict_file)
        else:
            l1 = Input(shape=(self.max_length,))
            l2 = Embedding(self.max_features, 128)(l1)
            l3 = SpatialDropout1D(0.2)(l2)
            l4 = LSTM(32, return_sequences=True, recurrent_dropout=0.2, dropout=0.2)(l3)
            l5 = LSTM(1, return_sequences=True, recurrent_dropout=0.2, dropout=0.2, activation='sigmoid')(l4)
            l6 = Lambda(lambda x: x[:, -1, :])(l5)

            self.model_predict = Model(inputs=[l1], outputs=[l5])
            self.model = Model(inputs=[l1], outputs=[l6])

            self.model.compile(metrics=['acc'], optimizer='adam', loss='binary_crossentropy')

            (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.max_features, skip_top=self.skip_top, seed=112)

            # Correct start char which is ruined by skip_top during load
            for i, y in zip(x_train, x_test):
                i[0] = 1
                y[0] = 1

            (x_train, x_test) = (self.pad(x_train), self.pad(x_test))

            self.model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,
                           callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
                           validation_data=(x_test, y_test))

            self.model.save(model_file)
            self.model_predict.save(model_predict_file)

    def read_sequence_from_file(self, filename):
        def get_mapping(word):
            if word in self.word_to_id.keys():
                mapping = self.word_to_id[word]
                if self.skip_top <= mapping < self.max_features:
                    return mapping
            # Out of vocabulary char
            return 2

        with open(filename) as file:
            word_seq = text_to_word_sequence(file.read())

        index_seq = [get_mapping(word) for word in word_seq]
        index_seq.insert(0, 1)

        return index_seq

    def benchmark_model(self, test_x, test_y):
        # Expects preprocessed test data
        print(f'Average accuracy of whole review sentiment prediction: {np.rint(self.model.evaluate(test_x, test_y)[1] * 100)}%')

    def generate_illustration(self, review_filename, output_filename):
        preprocessed_file = self.pad([self.read_sequence_from_file(review_filename)]).reshape(self.max_length)

        review_positive = True if self.model.predict(np.array([preprocessed_file])).reshape(1) >= 0.5 else False

        # Run prediction and get sigmoid activation vector with values 0.0 .. 1.0
        prediction_vector = list(self.model_predict.predict(np.array([preprocessed_file])).reshape(self.max_length))

        # Transform 0.0 .. 1.0 floats to 1 .. 510 ints for RGB values (255 red green)
        prediction_vector = np.interp(prediction_vector, [0, 1.0], [0, 510]).astype(int)

        preprocessed_file = list(preprocessed_file)

        # Find initial sentiment
        current_sentiment = prediction_vector[preprocessed_file.index(1) + 1]

        with open(review_filename) as origin_file:
            content = origin_file.read().split(' ')

        output_html = "<html>"

        # Match words from original text against prediction vector to update sentiment
        # Append word with color to HTML string
        for word in content:
            cleaned_word = self.cleanup_word(word)
            if cleaned_word in self.word_to_id.keys() and self.word_to_id[cleaned_word] in preprocessed_file:
                current_sentiment = prediction_vector[preprocessed_file.index(self.word_to_id[cleaned_word])]

            output_html += f'<span style="background-color:rgba({255 - max(current_sentiment - 255, 0)},' \
                f' {min(current_sentiment, 255)}, 0, 0.9);"> {word}</span>'

        output_html += f'<br><br> <b>review is: {"positive" if review_positive else "negative"}</b> </html>'

        with open(output_filename, "w") as output_file:
            output_file.write(output_html)

    def pad(self, sequences):
        return sequence.pad_sequences(sequences, maxlen=self.max_length, truncating='post')

    @staticmethod
    def cleanup_word(word):
        for c in '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n':
            word = word.replace(c, "")
        return word.lower()

    @staticmethod
    def get_imdb_word_mappings():
        index = {k: (v + 3) for k, v in imdb.get_word_index().items()}
        index["<PAD>"] = 0
        index["<START>"] = 1
        index["<UNK>"] = 2
        return index, {v: k for k, v in index.items()}


if __name__ == "__main__":
    imdbNN = ImdbNN(load=True)
    imdbNN.generate_illustration("imdb_review_sample.txt", "imdb_review_sample.html")

    (_, _), (x_test, y_test) = imdb.load_data(num_words=imdbNN.max_features, skip_top=imdbNN.skip_top, seed=1024)
    imdbNN.benchmark_model(imdbNN.pad(x_test[:5000]), y_test[:5000])


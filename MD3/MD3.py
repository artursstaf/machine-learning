from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.engine.saving import load_model
from keras.layers import Embedding, LSTM, Lambda, CuDNNLSTM, Dense, Flatten
from keras_preprocessing import sequence
from keras_preprocessing.text import text_to_word_sequence
from keras.datasets import imdb
import numpy as np


class Imdb:
    def __init__(self, load=False):
        (self.word_to_id, self.id_to_word) = self.get_imdb_word_mappings()
        self.max_features = 20000
        self.max_length = 400
        self.skip_top = 20

        if load:
            self.model = load_model("imdb_model.h5")
            self.model_predict = load_model("imdb_predict_model.h5")
            self.model.summary()
        else:
            L1 = Input(shape=(self.max_length,))
            L2 = Embedding(self.max_features, 50)(L1)
            L3 = CuDNNLSTM(128, return_sequences=True)(L2)
            L4 = CuDNNLSTM(64, return_sequences=True)(L3)
            L5 = CuDNNLSTM(1, return_sequences=True)(L4)
            L6 = Lambda(lambda x: x[:, -1, :])(L5)
            L8 = Dense(1, activation='sigmoid')(L6)

            self.model_predict = Model(inputs=[L1], outputs=[L5])
            self.model = Model(inputs=[L1], outputs=[L8])
            self.model.summary()

            self.model.compile(metrics=['acc'], optimizer='adam', loss='binary_crossentropy')

            (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz", num_words=self.max_features,
                                                                  skip_top=self.skip_top,
                                                                  seed=111,
                                                                  start_char=1, oov_char=2, index_from=3)
            # correct start char
            for i, y in zip(x_train, x_test):
                i[0] = 1
                y[0] = 1

            (x_train, x_test) = (self.pad(x_train), self.pad(x_test))

            self.model.fit(x_train, y_train, epochs=100, batch_size=128, verbose=1,
                           callbacks=[EarlyStopping(patience=2, restore_best_weights=True)],
                           validation_data=(x_test, y_test))

            # Save models separately
            self.model.save('imdb_model.h5')
            self.model_predict.save("imdb_predict_model.h5")

            print(self.model.evaluate(x_test, y_test))

    def read_review_from_file(self, filename):
        def get_mapping(word):
            if word in self.word_to_id.keys():
                mapping = self.word_to_id[word]
                # limit max words
                if self.skip_top <= mapping < self.max_features:
                    return mapping
            # oov character
            return 2

        with open(filename) as file:
            word_seq = text_to_word_sequence(file.read())
        index_seq = [get_mapping(word) for word in word_seq]
        index_seq.insert(0, 1)
        return index_seq

    def generate_illustration(self, review_filename, outout_filename):
        preprocessed_file = self.pad([self.read_review_from_file(review_filename)])[0]

        # Run prediction and get tanh activation -1.0 to 1.0
        prediction_vector = list(self.model_predict.predict(np.array([preprocessed_file])).reshape(self.max_length))

        # Transform -1.0 to 1.0 floats as 0 to 510 range for RGB values (red green)
        prediction_vector = np.interp(prediction_vector, [-1.0, 1.0], [0, 510]).astype(int)

        preprocessed_file = list(preprocessed_file)

        # find initial sentiment
        start_index = preprocessed_file.index(1) + 1
        current_sentiment = prediction_vector[start_index]

        with open(review_filename) as origin_file:
            content = origin_file.read().split(' ')

        output_str = "<html>"

        # match words from original text against prediction vector to update sentiment
        # append word with colored span
        for word in content:
            cleaned = self.cleanup_word(word)
            if cleaned in self.word_to_id.keys() and self.word_to_id[cleaned] in preprocessed_file:
                current_sentiment = prediction_vector[preprocessed_file.index(self.word_to_id[cleaned])]
            output_str += f'<span style="background-color:rgba({255 - current_sentiment % 256},' \
                f' {min(current_sentiment, 255)}, 0, 0.8);"> {word}</span>'

        output_str += "</html>"

        with open(outout_filename, "w") as output_html:
            output_html.write(output_str)

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
    imdb = Imdb(load=True)
    imdb.generate_illustration("imdb_review_sample.txt", "imdb_review_sample.html")

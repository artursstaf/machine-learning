import numpy
from keras.layers import Dense
from keras.models import Sequential

model = Sequential(
    [
        Dense(3, activation='sigmoid', input_dim=2, name='hidden'),
        Dense(6, activation='sigmoid'),
        Dense(1, activation='sigmoid'),
    ]
)

model.compile(optimizer='adam', loss='mean_squared_error')

dati = [
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0]
]


# and
atbildes = [
    0,
    1,
    1,
    1
]

#model.load_weights('or')
model.fit(numpy.array(dati), atbildes, epochs=10000, batch_size=1)

model.save_weights('or')
model.summary()
#print('svari {}'.format(model.get_weights()))

print(model.predict(numpy.array([[0, 1]]))[0][0])


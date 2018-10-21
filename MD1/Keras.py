import tensorflow as tf
from keras.layers import Dense, Activation
from keras.models import Sequential

model = Sequential()
model.add(Dense(32, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
# sgd scohcahstic gradient descent
# adam  adaptive moment estimation
# loss funkcija katrai problemai cita,
# squared error ja liela kluda tad liels zaudejums ^ 2
# absolute percentage error, logarithmic error, ! categorical_crossentropy, salidzina ar vectoru [0, 1, 0, 1, 1, 0]
# cosine_proximity vektors ar patvaligam vertibam [1.123123, 54645.34234, 23.2343]
model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae', 'accuracy'])


model.fit(datim atbildes, epochs=10000, batch_size=1, verbose=0)

# strukturai
#model.to_json() / model.from_json()

model.predit(dati)

model.evaluate(dati)

model.summary()

model.get_config() # struktura
model.get_weights() # parada svarus

#palasit par aktivacijas funckijam !! softmax
# svaru inicializacija
# truncatedNormal

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# Exercise 1
# Construire un MLP avec 2 couches cachées avec 200 neurones chacune et fonction d’activation de type ‘relu’ et une couche de sortie de type ‘softmax’.
# Ce réseau traite des entrées de dimension 400 et propose 10 valeurs en sortie.

# Init sequential model
model = Sequential()

model.add(Dense(200, input_shape=(400,), activation='relu'))

# Adding second hidden layer
model.add(Dense(200, activation='relu'))

# Adding out layer
model.add(Dense(10, activation='softmax'))

# Display model structure
model.summary()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.datasets import mnist

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

# Exercise 2
# Compiler le MLP en y associant une fonction de coût de type cross entropie catégorielle et un optimiseur de type descente stochastique du gradient (SDG).
# Il sera probablement nécessaire d’importer des packages de Keras.

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])

# Exercise 3
# Que signifie ces deux lignes ?
    # Trouvez la bonne valeur de nb_dimensions_entree.
    # Ces lignes permettent de transformer les images du jeu de données MNIST en un format que le MLP peut traiter.
# Chaque image dans MNIST est de taille 28x28 pixels, donc la valeur de nb_dimensions_entree est 28 * 28 = 784.
# Vous pouvez afficher le contenu des différentes structures de données impliquées ici.

nb_dimensions_input = 784

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("x_train dimension:", x_train.shape)      # (60000, 28, 28)
print("x_test dimensions:", x_test.shape)       # (10000, 28, 28)
print("y_train dimensions:", y_train.shape)     # (60000,)
print("y_test dimensions:", y_test.shape)       # (10000,)

x_train = x_train.reshape(60000, nb_dimensions_input)
x_test = x_test.reshape(10000, nb_dimensions_input)

print("x_train dimensions after reshaping:", x_train.shape)  # (60000, 784)
print("x_test dimensions after reshaping:", x_test.shape)    # (10000, 784)

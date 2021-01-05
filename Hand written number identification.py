# Importing libraries
import tensorflow as tf

#Downloading the Mnist Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#x_train.shape, y_train.shape, x_test.shape, y_test.shape
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train.shape
x_train.shape[0]
x_train.shape[1]
x_train.shape[2]

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train.shape
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test.shape

# Reshaping and Normalizing the Images
#input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# Building the Convolutional Neural Network
# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# Creating a Sequential Model and adding the layers
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation=tf.nn.softmax))

#Compiling and Fitting the Model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x=x_train,y=y_train, epochs=2) # Actual epochs = 10

#Evaluating the Model
model.evaluate(x_test, y_test)


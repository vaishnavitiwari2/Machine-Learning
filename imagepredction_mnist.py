import tensorflow as tf
#Keras has a number of built-in datasets that you can access with a single line of code like this.
data = tf.keras.datasets.fashion_mnist
(training_images,training_labels),(test_images,test_labels)=data.load_data()
#Normailizing the data so that every pixel is represented by a number between 0 & 1
training_images = training_images/255.0
test_images = test_images/255.0
#1-Sequential model to specify we have many layers. 2-Flatten isn't a layer of neurons but an input layer specification. 3-Our inputs are 28*28 images, but we want them to be treated as a series of numeric values. Flatten takes that square values(a 2D array) and turns it into a line(1D array). 4-The next one is a layer of neurons and we are specifying that we want 128 of them. This is called a hidden layer as it is hidden between the input and output layer and is not seen by the caller.5- The activation function is code that runs on each neuron. For the middle layer, most actively used one is relu as it returns a value only if it is greater than 0.There is another dense layer which is the output layer.This has 10 neurons becuase we have 10 labels. Each of these neurons end up with a probability that the given input pixel matches that class. We could loop through them to do that but the softmax function does that for us.
model=tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),tf.keras.layers.Dense(128,activation=tf.nn.relu),tf.keras.layers.Dense(10,activation=tf.nn.softmax)])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
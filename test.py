import tensorflow as tf
from tensorflow import keras
from keras_preprocessing import image
import numpy as np
import matplotlib.pyplot as plt 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
data = keras.datasets.fashion_mnist

(train_images, train_lables), (test_images, test_lables) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dense(10,activation="softmax")
])

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_lables, epochs=5)


#model = tf.keras.models.load_model("saved_model/my_model")

lugares = ("sudadera-hombre-cuello-redondo-orlando-gris-claro-frontal.png")

for i in lugares:
    images = image.load_img(i, target_size=(28, 28))
    x = image.img_to_array(images)
    x = tf.image.rgb_to_grayscale(x)
    x = np.expand_dims(x, axis=0)
    x = x/255.0
    prediccion = model.predict(x)
    print("Predicion: " + class_names[np.argmax(prediccion[0])])





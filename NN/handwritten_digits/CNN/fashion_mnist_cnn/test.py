import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = tf.keras.models.load_model('fashion_cnn_model.h5')

(_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# preprocessing
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

index = 2 # pick any index from the list 
sample = x_test[index].reshape(1, 28, 28, 1)
prediction = model.predict(sample)

predicted_class = np.argmax(prediction)
true_class = y_test[index]

plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
plt.title(f"Prediction: {class_names[predicted_class]}\nTrue: {class_names[true_class]}")
plt.axis('off')
plt.show()

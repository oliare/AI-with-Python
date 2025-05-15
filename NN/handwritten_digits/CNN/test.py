from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

### load model
model = load_model("num_cnn_model.h5")

### load custom image 
img_path = "NN/handwritten_digits/numbers/3.png"

### load, convert to grayscale, resize to 28x28
img = Image.open(img_path).convert("L").resize((28, 28))

### invert colors (white digit on black background)
img = np.invert(img)

### convert to array and normalize
img_array = np.array(img).astype("float32") / 255.0

### reshape to match input shape (1, 28, 28, 1)
img_array = img_array.reshape(1, 28, 28, 1)

prediction = model.predict(img_array)
predicted_label = np.argmax(prediction)

plt.imshow(img_array[0].reshape(28, 28), cmap="gray")
plt.title(f"Predicted Digit: {predicted_label}")
plt.axis("off")
plt.show()
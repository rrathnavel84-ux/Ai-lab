import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained AI model
model = tf.keras.models.load_model("waste_model.keras")

# Waste categories
categories = ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Trash"]

# Load waste image
image = Image.open("waste.jpg").convert("RGB")
image = image.resize((224, 224))

# Convert image into array
image_array = np.array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Predict waste category
prediction = model.predict(image_array)
result = categories[np.argmax(prediction)]

print("AI Waste Classification System")
print("Waste Category:", result)

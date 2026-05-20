import exporter
import pickle
import numpy as np
import os

# Load the saved model
with open("examples\output\my_saved_model.sav", "rb") as f:
    loaded_model = pickle.load(f)

# Prepare input for prediction
predict_input = np.array([[4]])

# Make prediction
result = loaded_model.predict(predict_input)

print(result[0])

from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to specific domain if needed)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Path to the saved .h5 model
model_path = "D:/AI Data Science/Deep Learning/Projects/Potato Disease Classification/saved_models/model_1.h5"

# Load the model with compile set to False for inference
try:
    model = load_model(model_path, compile=False)
    print("Model loaded successfully.")
except ValueError as e:
    print(f"Error loading model: {e}")

# Class names for prediction
CLASS_NAME = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello I am Waleed"

# Function to convert uploaded image to numpy array
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())

    # Preprocess image for prediction
    img_batch = np.expand_dims(image, 0)  # Expand dimensions to add batch size (1)

    # Get predictions using the loaded Keras model
    predictions = model(img_batch, training=False)
    
    predicted_class = CLASS_NAME[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)

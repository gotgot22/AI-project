import os
import json
import requests
import numpy as np
from io import BytesIO
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
import tensorflow_hub as hub

# Global model variable
model = None

@csrf_exempt
@require_http_methods(["POST"])
def predict_image_view(request):
    if request.method == 'POST':
        load_keras_model()
        data = json.loads(request.body)
        image_url = data.get('imageurl')

        if not image_url:
            return JsonResponse({'error': 'No image URL provided.'}, status=400)

        result = predict_image(image_url)
        return JsonResponse({'result': result})

    return JsonResponse({'error': 'Invalid request'}, status=400)

def adjust_keras(image_url):
    """Resize the input image to 224x224."""
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")  # Ensure RGB format
        image = image.resize((224, 224))  # Resize to 224x224
        image_array = np.array(image) / 255.0  # Normalize the image
        return np.expand_dims(image_array, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error in resizing image: {e}")
        return None

def predict_image(image_url):
    """Predict the class of the image."""
    try:
        resized_image_tensor = adjust_keras(image_url)
        if resized_image_tensor is None:
            return "Image resizing failed."

        predictions = model.predict(resized_image_tensor)
        class_labels = ["ripe", "unripe", "overripe", "underripe"]
        probabilities = {label: predictions[0][i] for i, label in enumerate(class_labels)}

        # Determine the predicted class based on the highest probability
        max_probability = max(probabilities.values())
        result = "unknown"

        if max_probability > 0.8:  # Check against threshold
            result = max(probabilities, key=probabilities.get)

        print("Predictions:", probabilities)
        print("Final prediction:", result)
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error during prediction"

def load_keras_model():
    """Load the pre-trained Keras model."""
    global model
    try:
        model_path = os.path.join('/path/to/your/model/model_Banana_model.h5')  # Update path
        model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@csrf_exempt
def upload_image_view(request):
    """Handle image upload and return the image URL."""
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        fs = FileSystemStorage(location='/path/to/your/static/images')  # Update path
        filename = fs.save(image.name, image)
        image_url = f"/static/images/{filename}"
        return JsonResponse({'image_url': request.build_absolute_uri(image_url)})

    return JsonResponse({'error': 'Invalid request'}, status=400)
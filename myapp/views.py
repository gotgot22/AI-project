import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from keras.models import load_model
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import numpy as np
import json
from django.conf import settings
from io import BytesIO
from django.core.files.storage import FileSystemStorage
import os
import requests
#import tensorflow_hub as hub
#from django.shortcuts import render
from django.http import JsonResponse
# from .static.model.model_Banana_model import load_keras_model, predict_image  # Import your functions
from django.core.files.storage import FileSystemStorage

# Ensure the model is loaded

@csrf_exempt
@require_http_methods(["POST"])
def predict_image_view(request):
       
    if request.method == 'POST':
        load_keras_model()
        data = json.loads(request.body)
        image_url = data.get('imageurl')
        
        
       

        # Load and preprocess the image
        # image = Image.open(uploaded_file)
        result = predict_image(image_url)  # Call your prediction function

        return JsonResponse({'result': result})

    return JsonResponse({'error': 'Invalid request'}, status=400)
    
def adjust_keras(image_url):
   # """Resize the input image to 224x224."""
  
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
        image = image.resize((224, 224))  # Resize to 224x224
        image_array = np.array(image)      # Convert to numpy array
        image_array = image_array / 255.0  # Normalize the image
        print("resize complete")
        return np.expand_dims(image_array, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error in resizing image: {e}")
        return None

def predict_image(image_tensor):
    try:
        # Resize the image to 224x224 before prediction
        resized_image_tensor = adjust_keras(image_tensor)
        if resized_image_tensor is None:
            return JsonResponse({'error': 'Image resizing failed.'}, status=500)
        else:
            print("resized!")

        predictions = model.predict(resized_image_tensor)
        predictions_overripe = predictions[0][0]
        predictions_ripe= predictions[0][1]
        predictions_rotten = predictions[0][2]
        predictions_unripe = predictions[0][3]

        threshold = 0.8
        print(predictions)

        if predictions_ripe > threshold:
            result = "ripe"
        elif predictions_unripe > threshold:
            result = "unripe"
        elif predictions_overripe > threshold:
            result = "overripe"
        elif predictions_rotten > threshold:
            result = "rotten"
        else:
            result = "unknown"

        return result
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    

def load_keras_model():
    global model 
    try:
        #model_path = os.path.join(settings.BASE_DIR, 'static', 'model', 'banana_model.h5')
        model_path = os.path.join('/Users/punyisa/Desktop/banana project/myproject/myapp/static/model/banana_model.h5')
        model = load_model(model_path)
        print("Model banana loaded successfully!")
        return True
    except Exception as e:
        print(f"Error: {e}")
        model = None
        return False
    

@csrf_exempt
def upload_image_view(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        fs = FileSystemStorage(location='/Users/punyisa/Desktop/banana project/myproject/myapp/static/images')
        filename = fs.save(image.name, image)
        image_url = f"/static/images/{filename}"

        return JsonResponse({'image_url': request.build_absolute_uri(image_url)})

    return JsonResponse({'error': 'Invalid request'}, status=400)

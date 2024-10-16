# myapp/urls.py
from .web import web
from django.urls import path
from .views import predict_image_view,upload_image_view  # Import your view function

urlpatterns = [
    path('predict/', predict_image_view, name='predict_image'),
    path('',web,name='web'),
    path('upload/',upload_image_view, name='upload_image_view'),
]

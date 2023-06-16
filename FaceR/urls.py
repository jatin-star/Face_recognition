# myproject/FaceR/urls.py

from django.urls import include, path

urlpatterns = [
    path('', include('myapp.urls')),

    # other URL patterns for your project
]

from django.urls import path
from . import views
from .views import home

urlpatterns = [
    path('', home, name='home'),
    path('index/', views.index, name='index'),
    path('capture_images', views.capture_images, name='capture_images'),
    path('generate-dataframe/', views.generate_dataframe, name='generate_dataframe'),
    path('add_face/', views.face_add, name='add_face'),
    path('prepare_images', views.prepare_images, name='prepare_images'),
    path('face_recognize', views.face_recognize_page, name='face_recognize'),
    path('recognize_fun', views.recognize_fun, name='recognize_fun'),
    path('data_loader', views.dataloader, name='data_loader'),
    path('database', views.data_base, name='database'),
    path('save_dataframe', views.save_dataframe, name='save_dataframe')
]

from django.urls import path
from .views import get_log

urlpatterns = [
    path('get-log/', get_log, name='get-log'),
]

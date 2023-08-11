from django.urls import path
from .views import get_log, get_completion, send_query

urlpatterns = [
    path('get-log/', get_log, name='get-log'),
    path('get-completion/', get_completion, name='get-completion'),
    path('send-query/', send_query, name='send-query')
]

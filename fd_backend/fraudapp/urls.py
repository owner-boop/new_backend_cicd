from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings
from .views import ChartsData , FileReaderAPIView , gpt , FraudPredictionAPIView,  InvokeAPIView

urlpatterns = [
    path("", views.home, name=""),
    path("api/piechart", ChartsData.as_view()),
    path("api/fileread/", FileReaderAPIView.as_view()),
    path("api/" , gpt.as_view()),
    path("api/predict/", FraudPredictionAPIView.as_view(), name='fraud-prediction'),
    path('invoke/', InvokeAPIView.as_view(), name='invoke'),
] + static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)

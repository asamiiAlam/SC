from django.urls import path
from . import views

app_name = "watch"

urlpatterns = [
    path("wearable-insights/", views.wearable_dashboard, name="wearable_dashboard"),
    path("api/wearable/ingest/", views.wearable_ingest, name="wearable_ingest"),
]
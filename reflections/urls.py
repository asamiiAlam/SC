from django.urls import path
from . import views

app_name = 'reflections'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('check-in/', views.daily_checkin, name='daily_checkin'),
    path('check-in/success/', views.checkin_success, name='checkin_success'),
    path('history/', views.checkin_history, name='checkin_history'),
    path('weekly/',views.weekly_reflection,name='weekly_reflection'),
    path('weekly/<int:offset>/',views.weekly_reflection,name='weekly_reflection_offset'),
    path('weekly/api/analyze/',views.weekly_reflection_api, name='weekly_reflection_api'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('profile/', views.profile, name='profile'),
   
]
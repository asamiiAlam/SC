


from django.urls import path

from . import views
from django.contrib import admin


urlpatterns = [
   path('register/',views.register, name='register'),
   path('login/',views.login, name='login'),
   path('logout/',views.logout, name='logout'),
   path('profile/',views.profile, name='profile'),
   path('activate/<uidb64>/<token>/', views.activate, name="activate"),
   path('forgotpassword/', views.forgot_password, name='forgot_password'),
   path('password-reset-confirm/<uidb64>/<token>/', views.password_reset_confirm, name='password_reset_confirm'),
]


from django.shortcuts import render

# Create your views here.
from django.contrib import messages

from django.contrib import auth
from django.shortcuts import redirect, render
from .forms import RegistrationForm
from .models import Account
from django.contrib.auth import authenticate, login as auth_login
from django.contrib.auth.decorators import login_required
from .utils import send_password_reset_email, send_verification_email
from django.urls import reverse
from django.contrib.auth.tokens import default_token_generator
from django.utils.encoding import force_str
from django.utils.http import urlsafe_base64_decode
from django.utils.http import urlsafe_base64_encode

# Create your views here.
def login(request):
    if request.method == 'POST':
            email = request.POST.get('email')
            password = request.POST.get('password')
            print(email,password)
            user = authenticate(request, email=email, password=password)
            if user is not None:
                auth_login(request, user)
                messages.success(request, 'You have successfully logged in.')
                return redirect('reflections:dashboard') 
            else:
               messages.error(request, 'Invalid email or password.')
               return redirect('login')
                
            
    return render(request, 'accounts/login.html')
def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            first_name = form.cleaned_data['first_name']
            last_name = form.cleaned_data['last_name']
            email = form.cleaned_data['email']
            phone_number = form.cleaned_data['phone_number']
            password = form.cleaned_data['password']
            user = Account.objects.create_user(
                email=email,
                username=email.split('@')[0],
                first_name=first_name,
                last_name=last_name,
                password=password
            )
            user.phone_number = phone_number
            user.save()
            send_verification_email(request, user)
            messages.success(request, 'Please check your email to verify your account before logging in.')
            return redirect('login')
    else:
        form = RegistrationForm()

    context={
        'form':form
    }
    return render(request, 'accounts/register.html',context)
@login_required
def logout(request):
    auth.logout(request)
    messages.success(request, 'You have successfully logged out.')
    return redirect('login')
def activate(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = Account.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, Account.DoesNotExist):
        user = None

    if user is not None and default_token_generator.check_token(user, token):
        user.is_active = True
        user.save(update_fields=["is_active"])
        messages.success(request, "Your account has been verified. You can now log in.")
        return redirect("login")

    messages.error(request, "Verification link is invalid or expired.")
    return redirect("register")
@login_required
def profile(request):
    
    return render(request, 'accounts/profile.html')

def forgot_password(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        try:
            user = Account.objects.get(email__iexact=email)
            send_password_reset_email(request, user)
            messages.success(request, 'A password reset link has been sent to your email.')
            return redirect('login')
        except Account.DoesNotExist:
            messages.error(request, 'No account found with that email address.')
            return redirect('forgot_password')
        
    return render(request, 'accounts/forgot_password.html')
def password_reset_confirm(request, uidb64, token):
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = Account.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, Account.DoesNotExist):
        user = None

    if user is not None and default_token_generator.check_token(user, token):
        if request.method == 'POST':
            new_password = request.POST.get('new_password')
            confirm_new_password = request.POST.get('confirm_new_password')

            if new_password != confirm_new_password:
                messages.error(request, 'Passwords do not match.')
                return redirect('password_reset_confirm', uidb64=uidb64, token=token)


            user.set_password(new_password)
            user.save() 
            messages.success(request, 'Your password has been reset successfully. You can now log in.')
            return redirect('login')
        return render(request, 'accounts/password_reset_confirm.html')
    else:
        messages.error(request, 'Password reset link is invalid or expired.')
        return redirect('forgot_password')
    

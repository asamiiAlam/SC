from django import forms
from django.forms import modelformset_factory
from django.utils import timezone
from Accounts.models import Account
from .models import DailyCheckIn, Deadline, STUDY_TIME_CHOICES, CONFIDENCE_CHOICES


class DailyCheckInForm(forms.ModelForm):
    study_time = forms.ChoiceField(
        choices=STUDY_TIME_CHOICES,
        widget=forms.HiddenInput(),
        required=True,
    )
    stress_level = forms.IntegerField(
        min_value=0,
        max_value=100,
        widget=forms.HiddenInput(),
        required=True,
        initial=50,
    )
    confidence_level = forms.ChoiceField(
        choices=CONFIDENCE_CHOICES,
        widget=forms.HiddenInput(),
        required=True,
    )
    reflection_note = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={
            'class': (
                'w-full bg-transparent border-none outline-none resize-none '
                'text-on-surface placeholder-on-surface-variant/50 text-base leading-relaxed'
            ),
            'rows': 5,
            'placeholder': "How are you feeling today? What's on your mind?",
        }),
    )

    class Meta:
        model = DailyCheckIn
        fields = ['study_time', 'stress_level', 'confidence_level', 'reflection_note']


class DeadlineForm(forms.ModelForm):
    title = forms.CharField(
        required=False,
        max_length=255,
        widget=forms.TextInput(attrs={
            'class': (
                'w-full pl-14 pr-5 py-4 rounded-xl bg-surface-container '
                'border border-surface-container-highest text-on-surface '
                'placeholder-on-surface-variant/50 outline-none focus:border-primary '
                'focus:ring-2 focus:ring-primary/20 transition-all'
            ),
            'placeholder': 'Deadline title…',
        }),
    )
    due_date = forms.DateField(
        required=False,
        widget=forms.DateInput(attrs={
            'type': 'date',
            'class': (
                'w-full pl-14 pr-5 py-4 rounded-xl bg-surface-container '
                'border border-surface-container-highest text-on-surface '
                'outline-none focus:border-primary focus:ring-2 focus:ring-primary/20 '
                'transition-all cursor-pointer'
            ),
        }),
    )

    class Meta:
        model = Deadline
        fields = ['title', 'due_date']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['due_date'].widget.attrs['min'] = timezone.localdate().isoformat()

    def clean(self):
        cleaned_data = super().clean()
        title = cleaned_data.get('title')
        due_date = cleaned_data.get('due_date')
        today = timezone.localdate()

        if title and not due_date:
            self.add_error('due_date', 'Please select a due date.')
        if due_date and not title:
            self.add_error('title', 'Please enter a deadline title.')
        if due_date and due_date < today:
            self.add_error('due_date', 'Due date cannot be earlier than today.')

        return cleaned_data


DeadlineFormSet = modelformset_factory(
    Deadline,
    form=DeadlineForm,
    extra=1,
    can_delete=False,
)

class UpdateProfileForm(forms.ModelForm):
    class Meta:
        model = Account
        fields = ['first_name', 'last_name', 'username', 'phone_number', 'email']
        widgets = {
            'first_name': forms.TextInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl px-6 py-4 text-on-surface focus:ring-2 focus:ring-primary/20 transition-all font-body text-lg',
                'placeholder': 'Enter first name',
            }),
            'last_name': forms.TextInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl px-6 py-4 text-on-surface focus:ring-2 focus:ring-primary/20 transition-all font-body text-lg',
                'placeholder': 'Enter last name',
            }),
            'username': forms.TextInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl pl-12 pr-6 py-4 text-on-surface focus:ring-2 focus:ring-primary/20 transition-all font-body text-lg',
                'placeholder': 'Enter username',
            }),
            'phone_number': forms.TextInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl pl-12 pr-6 py-4 text-on-surface focus:ring-2 focus:ring-primary/20 transition-all font-body text-lg',
                'placeholder': '+8801XXXXXXXXX',
            }),
            'email': forms.EmailInput(attrs={
                'class': 'w-full bg-surface-container-low border-none rounded-xl pl-12 pr-6 py-4 text-on-surface focus:ring-2 focus:ring-primary/20 transition-all font-body text-lg',
                'placeholder': 'email address',
            }),
        }

   
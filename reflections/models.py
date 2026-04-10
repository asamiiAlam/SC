from django.db import models
from django.conf import settings
from django.db.models import UniqueConstraint
from django.db.models.functions import TruncDate


STUDY_TIME_CHOICES = [
    ('lt1', '<1hr'),
    ('1to2', '1-2hrs'),
    ('2to4', '2-4hrs'),
    ('gt4', '4+hrs'),
]

CONFIDENCE_CHOICES = [
    ('very_low', 'Very Low'),
    ('low', 'Low'),
    ('medium', 'Medium'),
    ('high', 'High'),
]


class DailyCheckIn(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='check_ins',
    )
    study_time = models.CharField(max_length=10, choices=STUDY_TIME_CHOICES)
    stress_level = models.PositiveSmallIntegerField(help_text='0 (calm) to 100 (intense)')
    confidence_level = models.CharField(max_length=10, choices=CONFIDENCE_CHOICES)
    reflection_note = models.TextField(blank=True, default='')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Daily Check-In'
        verbose_name_plural = 'Daily Check-Ins'
        constraints = [
            UniqueConstraint(
                TruncDate('created_at'),
                'user',
                name='unique_checkin_per_user_per_day'
            )
        ]
       

    def __str__(self):
        return f'{self.user.email} — {self.created_at:%Y-%m-%d}'

    @property
    def stress_label(self):
        if self.stress_level <= 33:
            return 'Calm'
        elif self.stress_level <= 66:
            return 'Balanced'
        return 'Intense'


class Deadline(models.Model):
    checkin = models.ForeignKey(
        DailyCheckIn,
        on_delete=models.CASCADE,
        related_name='deadlines',
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='deadlines',
    )
    title = models.CharField(max_length=255)
    due_date = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['due_date']
        verbose_name = 'Deadline'
        verbose_name_plural = 'Deadlines'

    def __str__(self):
        return f'{self.title} — {self.due_date} ({self.user.email})'
class WeeklyReflection(models.Model):
    user                  = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='weekly_reflections')
    week_start            = models.DateField()
    week_end              = models.DateField()
 
    # Aggregated stats from daily check-ins
    avg_stress_level      = models.FloatField(default=0)
    avg_study_hours       = models.FloatField(default=0)
    dominant_confidence   = models.CharField(max_length=20, default='medium')
    total_deadlines       = models.IntegerField(default=0)
    completed_checkins    = models.IntegerField(default=0)
    user_summary          = models.TextField(blank=True, default='')
 
    # ML model outputs
    ml_burnout_risk           = models.CharField(max_length=20, blank=True, default='')
    ml_burnout_score          = models.FloatField(null=True, blank=True)
    ml_stress_trend           = models.CharField(max_length=20, blank=True, default='')
    ml_recommendation         = models.TextField(blank=True, default='')
    ml_focus_areas            = models.JSONField(default=list, blank=True)
    ml_predicted_next_stress  = models.FloatField(null=True, blank=True)
    ml_wellness_score         = models.FloatField(null=True, blank=True)
 
    created_at = models.DateTimeField(auto_now_add=True)
 
    class Meta:
        ordering = ['-week_start']
        unique_together = [('user', 'week_start')]
 
    def __str__(self):
        return f"{self.user.email} — week of {self.week_start}"


from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from .models import DailyCheckIn, Deadline,WeeklyReflection
@admin.register(DailyCheckIn)
class DailyCheckInAdmin(admin.ModelAdmin):
    list_display = ('user', 'study_time', 'stress_level', 'stress_label', 'confidence_level', 'created_at')
    list_filter = ('study_time', 'confidence_level', 'created_at')
    search_fields = ('user__email', 'user__username', 'reflection_note')
    readonly_fields = ('created_at',)
    date_hierarchy = 'created_at'
    ordering = ('-created_at',)
    
 
    fieldsets = (
        ('User', {'fields': ('user',)}),
        ('Check-In Data', {'fields': ('study_time', 'stress_level', 'confidence_level')}),
        ('Reflection', {'fields': ('reflection_note',)}),
        ('Metadata', {'fields': ('created_at',)}),
    )
 
    @admin.display(description='Stress Label')
    def stress_label(self, obj):
        return obj.stress_label
 
 
@admin.register(Deadline)
class DeadlineAdmin(admin.ModelAdmin):
    list_display = ('title', 'due_date', 'user', 'checkin', 'created_at')
    list_filter = ('due_date',)
    search_fields = ('title', 'user__email')
    ordering = ('due_date',)

@admin.register(WeeklyReflection)
class WeeklyReflectionAdmin(admin.ModelAdmin):
     list_display = ('user', 'week_start', 'created_at')
     list_filter = ('created_at',)
     search_fields = ('user__email',)
     ordering = ('-created_at',)

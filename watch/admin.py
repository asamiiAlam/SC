from django.contrib import admin
from .models import (
    WearableConnection,
    WearableDailySummary,
    WearableRawSample,
    WearableSyncLog,
)


@admin.register(WearableConnection)
class WearableConnectionAdmin(admin.ModelAdmin):
    list_display = ("user", "platform", "is_connected", "last_synced_at", "created_at")
    search_fields = ("user__email", "platform")


@admin.register(WearableDailySummary)
class WearableDailySummaryAdmin(admin.ModelAdmin):
    list_display = (
        "user", "sample_date", "steps", "distance_km", "calories_kcal",
        "avg_heart_rate", "sleep_minutes", "spo2"
    )
    search_fields = ("user__email",)
    list_filter = ("sample_date", "source")


@admin.register(WearableRawSample)
class WearableRawSampleAdmin(admin.ModelAdmin):
    list_display = ("user", "sample_type", "value", "unit", "start_time", "end_time", "source")
    search_fields = ("user__email", "sample_type")
    list_filter = ("sample_type", "source")


@admin.register(WearableSyncLog)
class WearableSyncLogAdmin(admin.ModelAdmin):
    list_display = ("user", "status", "platform", "synced_at")
    search_fields = ("user__email", "platform", "message")
    list_filter = ("status", "platform")
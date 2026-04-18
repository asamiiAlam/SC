from django.conf import settings
from django.db import models


class WearableConnection(models.Model):
    PLATFORM_CHOICES = [
        ("android_health_connect", "Android Health Connect"),
        ("apple_healthkit", "Apple HealthKit"),
    ]

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="wearable_connection"
    )
    platform = models.CharField(max_length=50, choices=PLATFORM_CHOICES)
    is_connected = models.BooleanField(default=False)
    last_synced_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.email} - {self.platform}"


class WearableDailySummary(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="wearable_daily_summaries"
    )
    sample_date = models.DateField()

    steps = models.IntegerField(default=0)
    distance_km = models.FloatField(null=True, blank=True)
    calories_kcal = models.FloatField(null=True, blank=True)
    avg_heart_rate = models.IntegerField(null=True, blank=True)
    resting_heart_rate = models.IntegerField(null=True, blank=True)
    sleep_minutes = models.IntegerField(null=True, blank=True)
    spo2 = models.FloatField(null=True, blank=True)

    source = models.CharField(max_length=50, default="mobile_bridge")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "sample_date")
        ordering = ["-sample_date"]

    def __str__(self):
        return f"{self.user.email} - {self.sample_date}"


class WearableRawSample(models.Model):
    SAMPLE_TYPES = [
        ("steps", "Steps"),
        ("heart_rate", "Heart Rate"),
        ("sleep", "Sleep"),
        ("spo2", "SpO2"),
        ("calories", "Calories"),
        ("distance", "Distance"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="wearable_raw_samples"
    )
    sample_type = models.CharField(max_length=30, choices=SAMPLE_TYPES)
    value = models.FloatField()
    unit = models.CharField(max_length=20, blank=True)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField(null=True, blank=True)
    source = models.CharField(max_length=50, default="mobile_bridge")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-start_time"]

    def __str__(self):
        return f"{self.user.email} - {self.sample_type} - {self.value}"


class WearableSyncLog(models.Model):
    STATUS_CHOICES = [
        ("success", "Success"),
        ("failed", "Failed"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="wearable_sync_logs"
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    platform = models.CharField(max_length=50)
    message = models.TextField(blank=True)
    synced_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-synced_at"]

    def __str__(self):
        return f"{self.user.email} - {self.status} - {self.synced_at}"
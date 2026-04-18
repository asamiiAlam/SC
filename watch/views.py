import json
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse, HttpResponseForbidden
from django.shortcuts import render
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from django.views.decorators.csrf import csrf_exempt

from .models import (
    WearableConnection,
    WearableDailySummary,
    WearableRawSample,
    WearableSyncLog,
)


@login_required
def wearable_dashboard(request):
    daily_summaries = WearableDailySummary.objects.filter(user=request.user).order_by("-sample_date")[:7]
    connection = WearableConnection.objects.filter(user=request.user).first()
    latest_summary = daily_summaries[0] if daily_summaries else None

    context = {
        "connection": connection,
        "latest_summary": latest_summary,
        "daily_summaries": reversed(daily_summaries),
    }
    return render(request, "wearable_dashboard.html", context)


@csrf_exempt
@login_required
def wearable_ingest(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method."}, status=405)

    try:
        payload = json.loads(request.body)

        platform = payload.get("platform", "unknown")
        daily_summaries = payload.get("daily_summaries", [])
        raw_samples = payload.get("raw_samples", [])

        connection, _ = WearableConnection.objects.get_or_create(
            user=request.user,
            defaults={
                "platform": platform,
                "is_connected": True,
                "last_synced_at": timezone.now(),
            }
        )

        connection.platform = platform
        connection.is_connected = True
        connection.last_synced_at = timezone.now()
        connection.save()

        for item in daily_summaries:
            WearableDailySummary.objects.update_or_create(
                user=request.user,
                sample_date=item["sample_date"],
                defaults={
                    "steps": item.get("steps", 0),
                    "distance_km": item.get("distance_km"),
                    "calories_kcal": item.get("calories_kcal"),
                    "avg_heart_rate": item.get("avg_heart_rate"),
                    "resting_heart_rate": item.get("resting_heart_rate"),
                    "sleep_minutes": item.get("sleep_minutes"),
                    "spo2": item.get("spo2"),
                    "source": item.get("source", "mobile_bridge"),
                }
            )

        for sample in raw_samples:
            start_time = parse_datetime(sample["start_time"])
            end_time = parse_datetime(sample["end_time"]) if sample.get("end_time") else None

            WearableRawSample.objects.create(
                user=request.user,
                sample_type=sample["sample_type"],
                value=sample["value"],
                unit=sample.get("unit", ""),
                start_time=start_time,
                end_time=end_time,
                source=sample.get("source", "mobile_bridge"),
            )

        WearableSyncLog.objects.create(
            user=request.user,
            status="success",
            platform=platform,
            message="Wearable data synced successfully."
        )

        return JsonResponse({"success": True, "message": "Wearable data synced successfully."})

    except Exception as e:
        WearableSyncLog.objects.create(
            user=request.user,
            status="failed",
            platform=payload.get("platform", "unknown") if "payload" in locals() else "unknown",
            message=str(e)
        )
        return JsonResponse({"error": str(e)}, status=400)
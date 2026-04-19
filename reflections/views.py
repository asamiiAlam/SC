from datetime import timedelta
from collections import Counter
import json
import logging
import os
from django.views.decorators.http import require_POST
from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.db.models import Avg
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.utils import timezone

from .forms import DailyCheckInForm, DeadlineFormSet, UpdateProfileForm
from .models import DailyCheckIn, Deadline, WeeklyReflection
from .stress_ml import predict_weekly_reflection

logger = logging.getLogger(__name__)

# Optional OpenAI integration
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

STUDY_TIME_HOURS = {
    'lt1': 0.5,
    '1to2': 1.5,
    '2to4': 3.0,
    'gt4': 5.0,
}


def _get_today_checkin(user):
    today = timezone.localdate()
    try:
        return DailyCheckIn.objects.prefetch_related('deadlines').get(
            user=user,
            created_at__date=today,
        )
    except DailyCheckIn.DoesNotExist:
        return None


def _get_week_bounds(ref_date=None):
    if ref_date is None:
        ref_date = timezone.localdate()
    monday = ref_date - timedelta(days=ref_date.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday


def _aggregate_checkins(checkins):
    if not checkins.exists():
        return {
            'avg_stress_level': 0,
            'avg_study_hours': 0,
            'dominant_confidence': 'medium',
            'total_deadlines': 0,
            'completed_checkins': 0,
        }

    avg_stress = checkins.aggregate(avg=Avg('stress_level'))['avg'] or 0
    study_hours_list = [STUDY_TIME_HOURS.get(c.study_time, 2.0) for c in checkins]
    avg_study = sum(study_hours_list) / len(study_hours_list)

    confidence_counts = Counter(c.confidence_level for c in checkins)
    dominant_confidence = confidence_counts.most_common(1)[0][0]

    checkin_ids = checkins.values_list('id', flat=True)
    total_deadlines = Deadline.objects.filter(checkin_id__in=checkin_ids).count()

    return {
        'avg_stress_level': round(avg_stress, 2),
        'avg_study_hours': round(avg_study, 2),
        'dominant_confidence': dominant_confidence,
        'total_deadlines': total_deadlines,
        'completed_checkins': checkins.count(),
    }


def _parse_focus_areas(weekly):
    if not weekly or not weekly.ml_focus_areas:
        return []

    if isinstance(weekly.ml_focus_areas, list):
        return weekly.ml_focus_areas

    try:
        areas = json.loads(weekly.ml_focus_areas)
        return areas if isinstance(areas, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _wellness_offset(score):
    try:
        pct = max(0.0, min(float(score), 100.0)) / 100.0
    except (TypeError, ValueError):
        pct = 0.0
    return round(188.0 * (1 - pct), 2)


def _clean_focus_areas(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()][:3]
    return []


def _build_ai_prompt(ml_input: dict, base_result: dict) -> str:
    return f"""
You are a supportive student wellness coach.

Write practical, supportive, non-clinical feedback in simple English.

Weekly student data:
- Average stress level: {ml_input.get('avg_stress_level')}
- Average study hours: {ml_input.get('avg_study_hours')}
- Dominant confidence: {ml_input.get('dominant_confidence')}
- Total deadlines: {ml_input.get('total_deadlines')}
- Completed check-ins: {ml_input.get('completed_checkins')}
- User summary: {ml_input.get('user_summary', '')}

Current app analysis:
- Burnout risk: {base_result.get('burnout_risk')}
- Burnout score: {base_result.get('burnout_score')}
- Stress trend: {base_result.get('stress_trend')}
- Predicted next stress: {base_result.get('predicted_next_stress')}
- Wellness score: {base_result.get('wellness_score')}

Return valid JSON only with exactly these keys:
recommendation
focus_areas
encouragement

Rules:
- recommendation: 1 short paragraph
- focus_areas: exactly 3 short strings
- encouragement: exactly 1 sentence
- do not diagnose any medical or mental health condition
- do not mention being an AI
""".strip()
def _generate_ai_feedback(ml_input: dict, base_result: dict) -> dict | None:
    """
    Optional OpenAI enhancement.
    Returns dict or None if OpenAI is unavailable/fails.
    """
    print("OPENAI_AVAILABLE =", OPENAI_AVAILABLE)
    print("OPENAI_API_KEY exists =", bool(os.getenv("OPENAI_API_KEY")))

    if not OPENAI_AVAILABLE:
        print("RETURNING: openai package not available")
        return None

    if not os.getenv("OPENAI_API_KEY"):
        print("RETURNING: OPENAI_API_KEY not found")
        return None

    try:
        print("OPENAI CALL STARTING")
        client = OpenAI()
        response = client.responses.create(
            model="gpt-5.4-mini",
            input=_build_ai_prompt(ml_input, base_result),
        )

        text = (response.output_text or "").strip()
        print("RAW OPENAI TEXT:", text)

        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()

        parsed = json.loads(text)

        recommendation = str(parsed.get("recommendation", "")).strip()
        focus_areas = _clean_focus_areas(parsed.get("focus_areas", []))
        encouragement = str(parsed.get("encouragement", "")).strip()

        if not recommendation or len(focus_areas) != 3:
            print("RETURNING: invalid parsed response")
            return None

        print("OPENAI SUCCESS")
        return {
            "recommendation": recommendation,
            "focus_areas": focus_areas,
            "encouragement": encouragement,
        }

    except Exception as exc:
        print("OPENAI ERROR:", exc)
        logger.exception("OpenAI feedback generation failed: %s", exc)
        return None

def _run_weekly_analysis(stats: dict, user_summary: str) -> dict:
    """
    Runs local ML/rule-based analysis first, then optionally enhances
    recommendation + focus areas with OpenAI.
    """
    ml_input = {
        'avg_stress_level': stats['avg_stress_level'],
        'avg_study_hours': stats['avg_study_hours'],
        'dominant_confidence': stats['dominant_confidence'],
        'total_deadlines': stats['total_deadlines'],
        'completed_checkins': stats['completed_checkins'],
        'user_summary': user_summary,
    }

    base_result = predict_weekly_reflection(ml_input)

    recommendation = base_result['recommendation']
    focus_areas = base_result['focus_areas']
    source_label = "LOCAL MODEL"

    ai_result = _generate_ai_feedback(ml_input, base_result)
    if ai_result:
        recommendation = ai_result['recommendation']
        focus_areas = ai_result['focus_areas']
        source_label = "OPENAI"

        encouragement = ai_result.get('encouragement', '')
        if encouragement:
            recommendation = f"{recommendation}\n\n{encouragement}"

    base_result['recommendation'] = f"[SOURCE: {source_label}]\n\n{recommendation}"
    base_result['focus_areas'] = focus_areas
    return base_result


# ─────────────────────────────────────────────────────────────────────────────
# dashboard
# ─────────────────────────────────────────────────────────────────────────────

@login_required
def dashboard(request):
    user = request.user
    today = timezone.localdate()
    week_start, week_end = _get_week_bounds(today)
    next_7 = today + timedelta(days=7)

    today_checkin = _get_today_checkin(user)

    latest_week = (
        WeeklyReflection.objects
        .filter(user=user)
        .order_by('-week_start')
        .first()
    )

    avg_stress = latest_week.avg_stress_level if latest_week else 0
    avg_study_hours = latest_week.avg_study_hours if latest_week else 0
    avg_study_hours_pct = (
        max(0, min(round((float(avg_study_hours) / 12) * 100), 100))
        if avg_study_hours else 0
    )

    deadline_qs = (
        Deadline.objects
        .filter(user=user, due_date__gte=today, due_date__lte=next_7)
        .order_by('due_date')
    )
    upcoming_deadlines_count = deadline_qs.count()

    upcoming_deadlines = [
        {
            'title': d.title,
            'due_date': d.due_date,
            'days_left': (d.due_date - today).days,
        }
        for d in deadline_qs
    ]

    weekly_checkins_qs = (
        DailyCheckIn.objects
        .filter(
            user=user,
            created_at__date__gte=week_start,
            created_at__date__lte=week_end,
        )
        .order_by('created_at')
    )

    checkins_this_week = min(weekly_checkins_qs.count(), 7)

    checkins_by_date = {}
    for c in weekly_checkins_qs:
        checkins_by_date[c.created_at.date()] = c

    recent_checkins = []
    for i in range(7):
        day = week_start + timedelta(days=i)
        checkin = checkins_by_date.get(day)

        if checkin:
            stress_level = int(checkin.stress_level)
            stress_pct = min(max(stress_level, 0), 100)
            has_data = True
            reflection_note = checkin.reflection_note
            study_time_display = checkin.get_study_time_display()
            confidence_display = checkin.get_confidence_level_display()
            confidence_level = checkin.confidence_level
            created_at = checkin.created_at
        else:
            stress_level = 0
            stress_pct = 0
            has_data = False
            reflection_note = ''
            study_time_display = ''
            confidence_display = ''
            confidence_level = ''
            created_at = None

        recent_checkins.append({
            'day_label': day.strftime('%a'),
            'date': day,
            'stress_level': stress_level,
            'stress_pct': stress_pct,
            'has_data': has_data,
            'reflection_note': reflection_note,
            'study_time_display': study_time_display,
            'confidence_display': confidence_display,
            'confidence_level': confidence_level,
            'created_at': created_at,
        })

    real_checkins = [c for c in recent_checkins if c['has_data']]
    if not latest_week and real_checkins:
        avg_stress = round(
            sum(c['stress_level'] for c in real_checkins) / len(real_checkins), 2
        )

        study_hours_list = [
            STUDY_TIME_HOURS.get(checkins_by_date[c['date']].study_time, 2.0)
            for c in real_checkins
        ]
        avg_study_hours = round(sum(study_hours_list) / len(study_hours_list), 2)
        avg_study_hours_pct = max(
            0, min(round((float(avg_study_hours) / 12) * 100), 100)
        )

    week_dots = range(1, 8)

    wellness_offset = _wellness_offset(
        latest_week.ml_wellness_score if latest_week else 0
    )
    focus_areas = _parse_focus_areas(latest_week)

    context = {
        'today_checkin': today_checkin,
        'recent_checkins': recent_checkins,
        'upcoming_deadlines': upcoming_deadlines,
        'avg_stress': avg_stress,
        'avg_study_hours': avg_study_hours,
        'avg_study_hours_pct': avg_study_hours_pct,
        'upcoming_deadlines_count': upcoming_deadlines_count,
        'checkins_this_week': checkins_this_week,
        'week_dots': week_dots,
        'latest_week': latest_week,
        'focus_areas': focus_areas,
        'wellness_offset': wellness_offset,
    }
    return render(request, 'reflections/dashboard.html', context)


# ─────────────────────────────────────────────────────────────────────────────
# daily check-in
# ─────────────────────────────────────────────────────────────────────────────

@login_required
def daily_checkin(request):
    today_checkin = _get_today_checkin(request.user)

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'add_deadline':
            if not today_checkin:
                form = DailyCheckInForm(request.POST)
                deadline_formset = DeadlineFormSet(
                    request.POST,
                    queryset=Deadline.objects.none(),
                    prefix='deadlines',
                )

                if not form.is_valid():
                    messages.error(
                        request,
                        'Please complete your check-in fields before adding a deadline.'
                    )
                    return render(request, 'reflections/daily_checkin.html', {
                        'form': form,
                        'deadline_formset': deadline_formset,
                        'today_checkin': None,
                    })

                today_checkin = form.save(commit=False)
                today_checkin.user = request.user
                today_checkin.save()

            deadline_formset = DeadlineFormSet(
                request.POST,
                queryset=Deadline.objects.none(),
                prefix='deadlines',
            )

            if deadline_formset.is_valid():
                saved = 0

                for deadline_form in deadline_formset:
                    cd = getattr(deadline_form, 'cleaned_data', None)
                    if not cd:
                        continue

                    title = cd.get('title')
                    due_date = cd.get('due_date')

                    if title and due_date:
                        deadline = deadline_form.save(commit=False)
                        deadline.user = request.user
                        deadline.checkin = today_checkin
                        deadline.save()
                        saved += 1

                if saved:
                    messages.success(request, f'{saved} deadline(s) saved.')
                else:
                    messages.warning(
                        request,
                        'Fill in a title and date to save a deadline.'
                    )

                return redirect('reflections:daily_checkin')

            messages.error(request, 'Please fix the errors in your deadlines.')
            return render(request, 'reflections/daily_checkin.html', {
                'form': DailyCheckInForm(instance=today_checkin),
                'deadline_formset': deadline_formset,
                'today_checkin': today_checkin,
            })

        if today_checkin:
            messages.warning(request, 'You have already checked in today.')
            return redirect('reflections:daily_checkin')

        form = DailyCheckInForm(request.POST)
        deadline_formset = DeadlineFormSet(
            request.POST,
            queryset=Deadline.objects.none(),
            prefix='deadlines',
        )

        if form.is_valid() and deadline_formset.is_valid():
            checkin = form.save(commit=False)
            checkin.user = request.user
            checkin.save()

            for deadline_form in deadline_formset:
                cd = getattr(deadline_form, 'cleaned_data', None)
                if not cd:
                    continue

                title = cd.get('title')
                due_date = cd.get('due_date')

                if title and due_date:
                    deadline = deadline_form.save(commit=False)
                    deadline.user = request.user
                    deadline.checkin = checkin
                    deadline.save()

            messages.success(
                request,
                'Check-in saved! Great job showing up for yourself today.'
            )
            return redirect('reflections:checkin_success')

        messages.error(request, 'Please fix the errors below.')

    else:
        form = DailyCheckInForm(instance=today_checkin) if today_checkin else DailyCheckInForm()
        deadline_formset = DeadlineFormSet(
            queryset=Deadline.objects.none(),
            prefix='deadlines',
        )

    context = {
        'form': form,
        'deadline_formset': deadline_formset,
        'today_checkin': today_checkin,
    }
    return render(request, 'reflections/daily_checkin.html', context)


# ─────────────────────────────────────────────────────────────────────────────
# check-in success
# ─────────────────────────────────────────────────────────────────────────────

@login_required
def checkin_success(request):
    today_checkin = _get_today_checkin(request.user)
    if not today_checkin:
        return redirect('reflections:daily_checkin')

    context = {'today_checkin': today_checkin}
    return render(request, 'reflections/checkin_success.html', context)


# ─────────────────────────────────────────────────────────────────────────────
# check-in history
# ─────────────────────────────────────────────────────────────────────────────

@login_required
def checkin_history(request):
    checkins = (
        DailyCheckIn.objects
        .filter(user=request.user)
        .prefetch_related('deadlines')
        .order_by('-created_at')
    )

    context = {'checkins': checkins}
    return render(request, 'reflections/checkin_history.html', context)


# ─────────────────────────────────────────────────────────────────────────────
# weekly reflection
# ─────────────────────────────────────────────────────────────────────────────
def _build_support_resources(reflection) -> dict:
    stress = float(reflection.avg_stress_level or 0)
    risk = (reflection.ml_burnout_risk or "").lower()
    confidence = (reflection.dominant_confidence or "").lower()

    breathing_videos = []
    wellbeing_videos = []
    exercises = []

    # Base breathing suggestions
    breathing_videos = [
        {
            "title": "5-Min Guided Box Breathing",
            "source": "YouTube",
            "url": "https://www.youtube.com/watch?v=aPYmZOhJF5Q",
            "embed_url": "https://www.youtube.com/embed/aPYmZOhJF5Q",
            "description": "A short 4-4-4-4 breathing exercise for stress and focus."
        },
        {
            "title": "Guided Progressive Muscle Relaxation",
            "source": "YouTube",
            "url": "https://www.youtube.com/watch?v=2IJUD-e14FY",
            "embed_url": "https://www.youtube.com/embed/2IJUD-e14FY",
            "description": "A guided body relaxation exercise to release physical tension."
        },
    ]

    # Base wellbeing suggestions
    wellbeing_videos = [
        {
            "title": "Daily Calm: 10 Minute Mindfulness Meditation",
            "source": "YouTube",
            "url": "https://www.youtube.com/watch?v=ZToicYcHIOU",
            "embed_url": "https://www.youtube.com/embed/ZToicYcHIOU",
            "description": "A calm guided meditation for presence and emotional reset."
        },
        {
            "title": "Headspace: 10-Minute Meditation for Stress",
            "source": "YouTube",
            "url": "https://www.youtube.com/watch?v=lS0kcSNlULw",
            "embed_url": "https://www.youtube.com/embed/lS0kcSNlULw",
            "description": "A short guided meditation for overthinking and stress."
        },
    ]

    # Rule-based support
    if risk == "high" or stress >= 70:
        exercises = [
            {
                "title": "Box Breathing",
                "duration": "2–5 min",
                "steps": [
                    "Breathe in for 4 seconds",
                    "Hold for 4 seconds",
                    "Breathe out for 4 seconds",
                    "Hold for 4 seconds and repeat 4 rounds",
                ],
            },
            {
                "title": "Shoulder Release Reset",
                "duration": "2 min",
                "steps": [
                    "Lift shoulders up gently",
                    "Hold for 3 seconds",
                    "Release slowly",
                    "Repeat 5 times with slow breathing",
                ],
            },
            {
                "title": "5-4-3-2-1 Grounding",
                "duration": "3 min",
                "steps": [
                    "Notice 5 things you can see",
                    "Notice 4 things you can feel",
                    "Notice 3 things you can hear",
                    "Notice 2 things you can smell",
                    "Notice 1 thing you can taste",
                ],
            },
        ]
    elif risk == "moderate" or 40 <= stress < 70:
        exercises = [
            {
                "title": "10-Min Mindfulness Pause",
                "duration": "10 min",
                "steps": [
                    "Sit comfortably",
                    "Focus on your breathing",
                    "Notice thoughts without judging them",
                    "Bring attention back to the breath each time",
                ],
            },
            {
                "title": "Progressive Muscle Reset",
                "duration": "5 min",
                "steps": [
                    "Tense hands for 5 seconds, then relax",
                    "Repeat with shoulders, jaw, and legs",
                    "Breathe slowly between each muscle group",
                ],
            },
            {
                "title": "Mini Study Recovery Break",
                "duration": "5 min",
                "steps": [
                    "Stand up and step away from your desk",
                    "Stretch your arms and neck",
                    "Take 6 slow breaths",
                    "Return to one small study task only",
                ],
            },
        ]
    else:
        exercises = [
            {
                "title": "Mindful Breathing Check-In",
                "duration": "2 min",
                "steps": [
                    "Pause your work",
                    "Take 5 slow breaths",
                    "Ask yourself how your energy feels",
                    "Choose the next task with intention",
                ],
            },
            {
                "title": "10-Min Mindfulness Session",
                "duration": "10 min",
                "steps": [
                    "Sit quietly",
                    "Focus on breathing and body sensations",
                    "Notice distractions gently",
                    "Return attention to the present moment",
                ],
            },
            {
                "title": "Leisure Protection Exercise",
                "duration": "5 min",
                "steps": [
                    "Write one non-study activity you will protect this week",
                    "Choose a day and time for it",
                    "Keep it small and realistic",
                ],
            },
        ]

    # Confidence tuning
    if confidence in ["low", "very_low"]:
        wellbeing_videos.append(
            {
                "title": "Goodful: 10-Minute Meditation For Stress",
                "source": "YouTube",
                "url": "https://www.youtube.com/watch?v=z6X5oEIg6Ak",
                "embed_url": "https://www.youtube.com/embed/z6X5oEIg6Ak",
                "description": "A short guided reset when you feel tense or discouraged."
            }
        )

    return {
        "breathing_videos": breathing_videos[:2],
        "wellbeing_videos": wellbeing_videos[:3],
        "exercises": exercises[:3],
    }
@login_required
def weekly_reflection(request, offset=0):
    today = timezone.localdate()
    offset = int(offset)
    ref_date = today - timedelta(weeks=offset)
    week_start, week_end = _get_week_bounds(ref_date)

    checkins = DailyCheckIn.objects.filter(
        user=request.user,
        created_at__date__range=(week_start, week_end),
    ).order_by('created_at')

    reflection, _ = WeeklyReflection.objects.get_or_create(
        user=request.user,
        week_start=week_start,
        defaults={'week_end': week_end},
    )

    stats = _aggregate_checkins(checkins)
    for field, value in stats.items():
        setattr(reflection, field, value)

    reflection.week_end = week_end

    if request.method == 'POST':
        action = request.POST.get('action', '')

        if action == 'save_summary':
            reflection.user_summary = request.POST.get('user_summary', '')
            reflection.save()
            messages.success(request, 'Weekly summary saved.')
            if offset == 0:
                return redirect('reflections:weekly_reflection')
            return redirect('reflections:weekly_reflection_offset', offset=offset)

        if action == 'run_ml':
            reflection.user_summary = request.POST.get(
                'user_summary',
                reflection.user_summary
            )

            result = _run_weekly_analysis(stats, reflection.user_summary)

            reflection.ml_burnout_risk = result['burnout_risk']
            reflection.ml_burnout_score = result['burnout_score']
            reflection.ml_stress_trend = result['stress_trend']
            reflection.ml_recommendation = result['recommendation']
            reflection.ml_focus_areas = result['focus_areas']
            reflection.ml_predicted_next_stress = result['predicted_next_stress']
            reflection.ml_wellness_score = result['wellness_score']
            reflection.save()

            if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
                messages.success(request, 'AI analysis complete!')
            else:
                messages.success(request, 'Analysis complete!')

            if offset == 0:
                return redirect('reflections:weekly_reflection')
            return redirect('reflections:weekly_reflection_offset', offset=offset)

    else:
        reflection.save(update_fields=[
            'avg_stress_level',
            'avg_study_hours',
            'dominant_confidence',
            'total_deadlines',
            'completed_checkins',
            'week_end',
        ])

    daily_labels = []
    daily_stress = []
    daily_study = []

    for i in range(7):
        day = week_start + timedelta(days=i)
        daily_labels.append(f"{day.strftime('%a')} {day.day}")

        day_qs = checkins.filter(created_at__date=day)
        if day_qs.exists():
            daily_stress.append(
                round(day_qs.aggregate(avg=Avg('stress_level'))['avg'] or 0, 1)
            )
            daily_study.append(
                round(
                    sum(STUDY_TIME_HOURS.get(c.study_time, 2) for c in day_qs) / day_qs.count(),
                    1
                )
            )
        else:
            daily_stress.append(None)
            daily_study.append(None)

    past_reflections = WeeklyReflection.objects.filter(
        user=request.user,
        week_start__lt=week_start,
    ).order_by('-week_start')[:4]

    checkin_ids = checkins.values_list('id', flat=True)
    deadlines = Deadline.objects.filter(
        checkin_id__in=checkin_ids
    ).order_by('due_date')

    support_resources = _build_support_resources(reflection)

    context = {
    'reflection': reflection,
    'week_start': week_start,
    'week_end': week_end,
    'checkins': checkins,
    'deadlines': deadlines,
    'daily_labels': json.dumps(daily_labels),
    'daily_stress': json.dumps(daily_stress),
    'daily_study': json.dumps(daily_study),
    'past_reflections': past_reflections,
    'offset': offset,
    'prev_offset': offset + 1,
    'next_offset': max(offset - 1, 0),
    'is_current_week': offset == 0,
    'has_ml_results': bool(reflection.ml_burnout_risk),
    'breathing_videos': support_resources['breathing_videos'],
    'wellbeing_videos': support_resources['wellbeing_videos'],
    'exercises': support_resources['exercises'],
    }
    return render(request, 'reflections/weekly_reflection.html', context)


@login_required
def weekly_reflection_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    try:
        body = json.loads(request.body or '{}')
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON body'}, status=400)

    week_start, week_end = _get_week_bounds()
    checkins = DailyCheckIn.objects.filter(
        user=request.user,
        created_at__date__range=(week_start, week_end),
    )

    stats = _aggregate_checkins(checkins)
    result = _run_weekly_analysis(stats, body.get('user_summary', ''))
    return JsonResponse(result)


@login_required
def profile(request):
    if request.method == 'POST':
        form = UpdateProfileForm(request.POST, instance=request.user)
        if form.is_valid():
            if not form.has_changed():
                messages.info(request, 'No changes to save.')
            else:
                form.save()
                messages.success(request, 'Your profile has been updated.')
            return redirect('reflections:profile')

        messages.error(request, 'Please correct the errors below.')
    else:
        form = UpdateProfileForm(instance=request.user)

    return render(request, 'reflections/profile.html', {'form': form})


@login_required
def change_password(request):
    if request.method == 'POST':
        old_password = request.POST.get('old_password', '').strip()
        new_password = request.POST.get('new_password', '').strip()
        confirm_password = request.POST.get('confirm_password', '').strip()

        if not old_password or not new_password or not confirm_password:
            messages.error(request, 'All password fields are required.')
        elif not request.user.check_password(old_password):
            messages.error(request, 'Old password is incorrect.')
        elif new_password != confirm_password:
            messages.error(request, 'New password and confirmation do not match.')
        elif len(new_password) < 8:
            messages.error(request, 'New password must be at least 8 characters long.')
        else:
            request.user.set_password(new_password)
            request.user.save()
            update_session_auth_hash(request, request.user)
            messages.success(request, 'Your password has been changed successfully.')
            return redirect('reflections:profile')

    return render(request, 'reflections/change_password.html')
def _build_chatbot_prompt(user_message: str, user, dashboard_context: dict) -> str:
    first_name = user.first_name or user.username or "Student"

    avg_stress = dashboard_context.get("avg_stress", 0)
    avg_study_hours = dashboard_context.get("avg_study_hours", 0)
    upcoming_deadlines_count = dashboard_context.get("upcoming_deadlines_count", 0)
    checkins_this_week = dashboard_context.get("checkins_this_week", 0)

    return f"""
You are a supportive mental wellbeing chatbot for a student reflection app.

User name: {first_name}
Current dashboard summary:
- Average stress this week: {avg_stress}/100
- Average study hours: {avg_study_hours}
- Upcoming deadlines in next 7 days: {upcoming_deadlines_count}
- Check-ins completed this week: {checkins_this_week}

User message:
{user_message}

Rules:
- Reply in simple, warm, supportive English.
- Keep the response practical and short.
- Give emotional support, calming suggestions, study-balance tips, and gentle coping ideas.
- Do not diagnose any mental illness.
- Do not claim to be a therapist or doctor.
- If the user mentions self-harm, suicide, wanting to die, or being unsafe, tell them to contact emergency services, a trusted person, or a mental health professional immediately.
- Keep the response around 80 to 140 words.
- Do not say “as an AI”.
""".strip()


def _local_chatbot_reply(user_message: str, dashboard_context: dict) -> str:
    msg = (user_message or "").lower().strip()
    avg_stress = dashboard_context.get("avg_stress", 0)
    upcoming_deadlines_count = dashboard_context.get("upcoming_deadlines_count", 0)

    crisis_words = [
        "suicide", "kill myself", "want to die", "self-harm",
        "hurt myself", "end my life", "not safe"
    ]
    if any(word in msg for word in crisis_words):
        return (
            "I’m really sorry you’re feeling this way. Please contact emergency services, "
            "a trusted person, or a mental health professional right now. You should not stay alone with this. "
            "If possible, call someone immediately and tell them you need urgent support."
        )

    if any(word in msg for word in ["stress", "stressed", "pressure", "overwhelmed"]):
        if upcoming_deadlines_count > 0:
            return (
                "It sounds like pressure may be building up, especially with upcoming deadlines. "
                "Try choosing just one small task for the next 10 to 15 minutes instead of thinking about everything at once. "
                "Take one slow breath in, exhale longer than you inhale, and remind yourself that steady progress is enough for today."
            )
        return (
            "It sounds like you’re feeling overwhelmed. Try pausing for two minutes, relaxing your shoulders, "
            "and picking one very small next step. You do not need to solve the whole day at once. "
            "A gentle routine and short breaks can help your mind settle."
        )

    if any(word in msg for word in ["sad", "down", "low", "empty", "upset"]):
        return (
            "I’m sorry you’re feeling low. Try to be extra gentle with yourself today. "
            "A small reset may help: drink water, get a little fresh air, or message someone you trust. "
            "You do not have to carry everything alone."
        )

    if any(word in msg for word in ["anxious", "anxiety", "panic", "worried", "nervous"]):
        return (
            "When anxiety rises, it can help to slow your body first. "
            "Try breathing in for 4 seconds, hold for 4, and breathe out for 6. Repeat that a few times. "
            "Then focus only on the next simple action, not the whole future."
        )

    if any(word in msg for word in ["focus", "concentrate", "distracted", "motivation", "lazy"]):
        return (
            "If focus feels hard, make the task smaller. Study for only 10 minutes, keep your phone away, "
            "and aim to finish one tiny piece. Starting small is often better than waiting to feel fully ready."
        )

    if any(word in msg for word in ["sleep", "tired", "exhausted", "fatigue", "burnout"]):
        return (
            "You may be carrying too much right now. Rest is not wasted time. "
            "Try reducing pressure for the next few hours, avoid heavy multitasking, and create a calmer evening routine if you can. "
            "A rested mind usually works better than a forced one."
        )

    if avg_stress >= 70:
        return (
            "Your recent pattern suggests this may be a high-pressure week. "
            "Try protecting your energy with shorter study blocks, clearer priorities, and a few breathing pauses during the day. "
            "Doing less, but with calm focus, may help more than pushing yourself too hard."
        )

    return (
        "Thank you for sharing that. Take a slow breath and check what you need most right now: rest, structure, or encouragement. "
        "Pick one small caring action for yourself today, and let that be enough for this moment."
    )
def _get_music_suggestion(user_message: str, dashboard_context: dict) -> dict:
    msg = (user_message or "").lower().strip()
    avg_stress = dashboard_context.get("avg_stress", 0)

    if any(word in msg for word in ["stress", "stressed", "pressure", "overwhelmed"]):
        return {
            "music_type": "Calm",
            "music_title": "Calm Piano",
            "music_description": "Soft piano for stress relief",
            "music_file": "music/calm_piano.mp3",
        }

    if any(word in msg for word in ["anxious", "anxiety", "panic", "worried", "nervous"]):
        return {
            "music_type": "Breathing",
            "music_title": "Breathing Ambience",
            "music_description": "Gentle rain sound for calming down",
            "music_file": "music/breathing_ambience.mp3",
        }

    if any(word in msg for word in ["focus", "concentrate", "distracted", "motivation"]):
        return {
            "music_type": "Focus",
            "music_title": "Rain Focus",
            "music_description": "Relaxing rain for concentration",
            "music_file": "music/rain_focus.mp3",
        }

    if any(word in msg for word in ["tired", "sleep", "exhausted", "fatigue", "burnout"]):
        return {
            "music_type": "Rest",
            "music_title": "Soft Ambient Reset",
            "music_description": "Light ambient sound for rest and reset",
            "music_file": "music/soft_ambient.mp3",
        }

    if avg_stress >= 70:
        return {
            "music_type": "Calm",
            "music_title": "Peaceful Piano",
            "music_description": "Gentle piano for high-stress moments",
            "music_file": "music/peaceful_piano.mp3",
        }

    return {
        "music_type": "Study",
        "music_title": "Deep Focus",
        "music_description": "Steady background music for studying",
        "music_file": "music/deep_focus.mp3",
    }
@login_required
@require_POST
def mental_health_chat_api(request):
    try:
        data = json.loads(request.body or "{}")
        user_message = str(data.get("message", "")).strip()

        if not user_message:
            return JsonResponse({
                "reply": "Please share a little about how you are feeling."
            }, status=400)

        user = request.user
        today = timezone.localdate()
        week_start, week_end = _get_week_bounds(today)
        next_7 = today + timedelta(days=7)

        latest_week = (
            WeeklyReflection.objects
            .filter(user=user)
            .order_by("-week_start")
            .first()
        )

        avg_stress = latest_week.avg_stress_level if latest_week else 0
        avg_study_hours = latest_week.avg_study_hours if latest_week else 0

        deadline_qs = (
            Deadline.objects
            .filter(user=user, due_date__gte=today, due_date__lte=next_7)
            .order_by("due_date")
        )

        weekly_checkins_qs = (
            DailyCheckIn.objects
            .filter(
                user=user,
                created_at__date__gte=week_start,
                created_at__date__lte=week_end,
            )
            .order_by("created_at")
        )

        if not latest_week and weekly_checkins_qs.exists():
            avg_stress = round(
                weekly_checkins_qs.aggregate(avg=Avg("stress_level"))["avg"] or 0, 2
            )

            study_hours_list = [
                STUDY_TIME_HOURS.get(c.study_time, 2.0)
                for c in weekly_checkins_qs
            ]
            avg_study_hours = round(sum(study_hours_list) / len(study_hours_list), 2) if study_hours_list else 0

        dashboard_context = {
            "avg_stress": avg_stress,
            "avg_study_hours": avg_study_hours,
            "upcoming_deadlines_count": deadline_qs.count(),
            "checkins_this_week": min(weekly_checkins_qs.count(), 7),
        }

        reply = None

        crisis_words = [
            "suicide", "kill myself", "want to die", "self-harm",
            "hurt myself", "end my life", "not safe"
        ]
        is_crisis = any(word in user_message.lower() for word in crisis_words)

        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                client = OpenAI()
                response = client.responses.create(
                    model="gpt-5.4-mini",
                    input=_build_chatbot_prompt(user_message, user, dashboard_context),
                )
                reply = (response.output_text or "").strip()
            except Exception as exc:
                logger.exception("Mental health chatbot OpenAI error: %s", exc)
                reply = None

        if not reply:
            reply = _local_chatbot_reply(user_message, dashboard_context)
        music_suggestion = _get_music_suggestion(user_message, dashboard_context)

        return JsonResponse({
            "reply": reply,
            "is_crisis": is_crisis,
            "music_suggestion": music_suggestion,
        })

    except Exception as exc:
        return JsonResponse({
        "reply": "Sorry, something went wrong. Please try again.",
          "is_crisis": False,
         "music_suggestion": None,
         }, status=500)
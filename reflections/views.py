from datetime import date, timedelta
from collections import Counter
import json

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Avg
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.utils import timezone
from httpcore import request

from .forms import DailyCheckInForm, DeadlineFormSet,UpdateProfileForm
from .models import DailyCheckIn, Deadline, WeeklyReflection
from .stress_ml import predict_weekly_reflection


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

    # Weekly reflection stats
    avg_stress = latest_week.avg_stress_level if latest_week else 0
    avg_study_hours = latest_week.avg_study_hours if latest_week else 0
    avg_study_hours_pct = (
        max(0, min(round((float(avg_study_hours) / 12) * 100), 100))
        if avg_study_hours else 0
    )

    # Upcoming deadlines
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

    # Current week's actual check-ins
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

    # Map by date so dashboard can always show Mon-Sun
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

    # Fallback weekly averages if no weekly reflection yet
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

        # -------------------------------------------------
        # Add deadline(s)
        # -------------------------------------------------
        if action == 'add_deadline':
            if not today_checkin:
                form = DailyCheckInForm(request.POST)
                deadline_formset = DeadlineFormSet(
                    request.POST,
                    queryset=Deadline.objects.none(),
                    prefix='deadlines',
                )

                if not form.is_valid():
                    messages.error(request, 'Please complete your check-in fields before adding a deadline.')
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
                    messages.warning(request, 'Fill in a title and date to save a deadline.')

                return redirect('reflections:daily_checkin')

            messages.error(request, 'Please fix the errors in your deadlines.')
            return render(request, 'reflections/daily_checkin.html', {
                'form': DailyCheckInForm(instance=today_checkin),
                'deadline_formset': deadline_formset,
                'today_checkin': today_checkin,
            })

        # -------------------------------------------------
        # Submit full daily check-in
        # -------------------------------------------------
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

            messages.success(request, 'Check-in saved! Great job showing up for yourself today.')
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
            reflection.user_summary = request.POST.get('user_summary', reflection.user_summary)

            ml_input = {
                'avg_stress_level': stats['avg_stress_level'],
                'avg_study_hours': stats['avg_study_hours'],
                'dominant_confidence': stats['dominant_confidence'],
                'total_deadlines': stats['total_deadlines'],
                'completed_checkins': stats['completed_checkins'],
                'user_summary': reflection.user_summary,
            }

            result = predict_weekly_reflection(ml_input)

            reflection.ml_burnout_risk = result['burnout_risk']
            reflection.ml_burnout_score = result['burnout_score']
            reflection.ml_stress_trend = result['stress_trend']
            reflection.ml_recommendation = result['recommendation']
            reflection.ml_focus_areas = result['focus_areas']
            reflection.ml_predicted_next_stress = result['predicted_next_stress']
            reflection.ml_wellness_score = result['wellness_score']
            reflection.save()

            messages.success(request, 'AI analysis complete!')
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
                round(sum(STUDY_TIME_HOURS.get(c.study_time, 2) for c in day_qs) / day_qs.count(), 1)
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
    }
    return render(request, 'reflections/weekly_reflection.html', context)


@login_required
def weekly_reflection_api(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST only'}, status=405)

    body = json.loads(request.body)

    week_start, week_end = _get_week_bounds()
    checkins = DailyCheckIn.objects.filter(
        user=request.user,
        created_at__date__range=(week_start, week_end),
    )

    stats = _aggregate_checkins(checkins)
    stats['user_summary'] = body.get('user_summary', '')

    result = predict_weekly_reflection(stats)
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
        else:
            messages.error(request, 'Please correct the errors below.')
     else:
        form = UpdateProfileForm(instance=request.user)
        return render(request, 'reflections/profile.html', {'form': form})
@login_required
def change_password(request):
    if request.method == 'POST':
        old_password = request.POST.get('old_password')
        new_password = request.POST.get('new_password')
        confirm_password = request.POST.get('confirm_password')
        if not request.user.check_password(old_password):
            messages.error(request, 'Old password is incorrect.')
        elif new_password != confirm_password:
            messages.error(request, 'New password and confirmation do not match.')
     
        else:
            request.user.set_password(new_password)
            request.user.save()
            messages.success(request, 'Your password has been changed. Please log in again.')
            return redirect('login')
    
    return render(request, 'reflections/change_password.html')
    
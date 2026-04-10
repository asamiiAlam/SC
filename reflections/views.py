from datetime import date, timedelta
from collections import Counter
import json

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db.models import Avg
from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.utils import timezone

from .forms import DailyCheckInForm, DeadlineFormSet
from .models import DailyCheckIn, Deadline, WeeklyReflection
from .stress_ml import predict_weekly_reflection


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

STUDY_TIME_HOURS = {
    '0to1': 0.5,
    '1to2': 1.5,
    '2to4': 3.0,
    '4to6': 5.0,
    '6plus': 7.0,
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
        ref_date = date.today()
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


# ─────────────────────────────────────────────────────────────────────────────
# dashboard
# ─────────────────────────────────────────────────────────────────────────────

@login_required
def dashboard(request):
    today_checkin = _get_today_checkin(request.user)

    recent_checkins = (
        DailyCheckIn.objects
        .filter(user=request.user)
        .prefetch_related('deadlines')
        .order_by('-created_at')[:7]
    )

    upcoming_deadlines = (
        Deadline.objects
        .filter(user=request.user, due_date__gte=timezone.localdate())
        .order_by('due_date')[:5]
    )

    context = {
        'today_checkin': today_checkin,
        'recent_checkins': recent_checkins,
        'upcoming_deadlines': upcoming_deadlines,
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
                checkin_form = DailyCheckInForm(request.POST)
                if not checkin_form.is_valid():
                    deadline_formset = DeadlineFormSet(request.POST, prefix='deadlines')
                    messages.error(request, 'Please complete your check-in fields before adding a deadline.')
                    context = {
                        'form': checkin_form,
                        'deadline_formset': deadline_formset,
                        'today_checkin': None,
                    }
                    return render(request, 'reflections/daily_checkin.html', context)

                today_checkin = checkin_form.save(commit=False)
                today_checkin.user = request.user
                today_checkin.save()

            deadline_formset = DeadlineFormSet(
                request.POST,
                instance=today_checkin,
                prefix='deadlines',
            )

            if deadline_formset.is_valid():
                saved = 0
                for dl_form in deadline_formset:
                    cd = dl_form.cleaned_data
                    if cd.get('title') and cd.get('due_date') and not dl_form.instance.pk:
                        dl = dl_form.save(commit=False)
                        dl.user = request.user
                        dl.checkin = today_checkin
                        dl.save()
                        saved += 1

                if saved:
                    messages.success(request, f'{saved} deadline(s) saved.')
                else:
                    messages.warning(request, 'Fill in a title and date to save a deadline.')
            else:
                messages.error(request, 'Please fix the errors in your deadlines.')

            return redirect('reflections:daily_checkin')

        if today_checkin:
            messages.warning(request, 'You have already checked in today.')
            return redirect('reflections:daily_checkin')

        form = DailyCheckInForm(request.POST)
        deadline_formset = DeadlineFormSet(request.POST, prefix='deadlines')

        if form.is_valid() and deadline_formset.is_valid():
            checkin = form.save(commit=False)
            checkin.user = request.user
            checkin.save()

            for deadline_form in deadline_formset:
                cd = deadline_form.cleaned_data
                if cd.get('title') and cd.get('due_date'):
                    dl = deadline_form.save(commit=False)
                    dl.user = request.user
                    dl.checkin = checkin
                    dl.save()

            messages.success(request, 'Check-in saved! Great job showing up for yourself today.')
            return redirect('reflections:checkin_success')

        messages.error(request, 'Please fix the errors below.')

    else:
        if today_checkin:
            form = DailyCheckInForm(instance=today_checkin)
            deadline_formset = DeadlineFormSet(instance=today_checkin, prefix='deadlines')
        else:
            form = DailyCheckInForm()
            deadline_formset = DeadlineFormSet(prefix='deadlines')

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
    today = date.today()
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
def dashboard(request):
    return render(request, 'dashboard.html')
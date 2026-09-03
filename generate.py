import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)
N = 150000

print(f"Generating synthetic dataset with {N:,} rows...")

# -----------------------------------------------------------------------------
# 1. DEMOGRAPHICS & PERSONA ASSIGNMENT
# -----------------------------------------------------------------------------
personas = [
    'Digital Professional', 
    'Social Scroller', 
    'Night Owl', 
    'Balanced User', 
    'Entertainment Heavy'
]
persona_probs = [0.25, 0.22, 0.18, 0.20, 0.15]
assigned_persona = np.random.choice(personas, size=N, p=persona_probs)

user_ids = [f"USR_{i:06d}" for i in range(1, N + 1)]
ages = np.random.randint(18, 66, size=N)
genders = np.random.choice(['Male', 'Female', 'Non-Binary', 'Prefer not to say'], size=N, p=[0.48, 0.48, 0.02, 0.02])
occupations = np.random.choice(['Software/Tech', 'Corporate/Office', 'Student', 'Creative/Design', 'Healthcare', 'Education', 'Other'], size=N)
work_schedules = np.random.choice(['Standard (9-5)', 'Shift Work', 'Flexible', 'Freelance'], size=N, p=[0.55, 0.15, 0.20, 0.10])
education_levels = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], size=N, p=[0.25, 0.50, 0.20, 0.05])

weekdays = np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], size=N)
is_weekend = np.where(np.isin(weekdays, ['Saturday', 'Sunday']), 1, 0)
remote_work = np.where(is_weekend == 1, 0, np.random.choice([0, 1], size=N, p=[0.6, 0.4]))
exam_period = np.random.choice([0, 1], size=N, p=[0.92, 0.08])
holiday_period = np.random.choice([0, 1], size=N, p=[0.90, 0.10])

# -----------------------------------------------------------------------------
# 2. FEATURE GENERATION BASED ON LATENT PERSONAS
# -----------------------------------------------------------------------------
comp_hours = np.zeros(N)
smart_hours = np.zeros(N)
tab_hours = np.zeros(N)
social_hours = np.zeros(N)
stream_hours = np.zeros(N)
game_hours = np.zeros(N)
prod_hours = np.zeros(N)
night_hours = np.zeros(N)
unlocks = np.zeros(N, dtype=int)
notifications = np.zeros(N, dtype=int)
social_sessions = np.zeros(N, dtype=int)
sleep_hours = np.zeros(N)
physical_activity = np.zeros(N)
outdoor_hours = np.zeros(N)

for p in personas:
    idx = (assigned_persona == p)
    count = np.sum(idx)
    
    if p == 'Digital Professional':
        comp_hours[idx] = np.random.normal(7.5, 1.2, count)
        smart_hours[idx] = np.random.normal(3.0, 0.8, count)
        tab_hours[idx] = np.random.normal(0.8, 0.4, count)
        social_hours[idx] = np.random.normal(1.5, 0.5, count)
        stream_hours[idx] = np.random.normal(1.5, 0.6, count)
        game_hours[idx] = np.random.normal(0.5, 0.3, count)
        prod_hours[idx] = np.random.normal(6.0, 1.0, count)
        night_hours[idx] = np.random.normal(0.8, 0.4, count)
        unlocks[idx] = np.random.poisson(55, count)
        notifications[idx] = np.random.poisson(110, count)
        social_sessions[idx] = np.random.poisson(18, count)
        sleep_hours[idx] = np.random.normal(7.2, 0.7, count)
        physical_activity[idx] = np.random.normal(35, 12, count)
        outdoor_hours[idx] = np.random.normal(1.2, 0.4, count)

    elif p == 'Social Scroller':
        comp_hours[idx] = np.random.normal(2.5, 1.0, count)
        smart_hours[idx] = np.random.normal(7.0, 1.5, count)
        tab_hours[idx] = np.random.normal(1.2, 0.5, count)
        social_hours[idx] = np.random.normal(5.5, 1.2, count)
        stream_hours[idx] = np.random.normal(2.0, 0.8, count)
        game_hours[idx] = np.random.normal(0.8, 0.5, count)
        prod_hours[idx] = np.random.normal(1.2, 0.5, count)
        night_hours[idx] = np.random.normal(2.2, 0.8, count)
        unlocks[idx] = np.random.poisson(130, count)
        notifications[idx] = np.random.poisson(210, count)
        social_sessions[idx] = np.random.poisson(65, count)
        sleep_hours[idx] = np.random.normal(6.3, 1.0, count)
        physical_activity[idx] = np.random.normal(25, 10, count)
        outdoor_hours[idx] = np.random.normal(0.9, 0.4, count)

    elif p == 'Night Owl':
        comp_hours[idx] = np.random.normal(4.0, 1.5, count)
        smart_hours[idx] = np.random.normal(5.5, 1.2, count)
        tab_hours[idx] = np.random.normal(1.0, 0.5, count)
        social_hours[idx] = np.random.normal(3.2, 1.0, count)
        stream_hours[idx] = np.random.normal(3.5, 1.0, count)
        game_hours[idx] = np.random.normal(2.5, 1.2, count)
        prod_hours[idx] = np.random.normal(2.0, 0.8, count)
        night_hours[idx] = np.random.normal(4.2, 0.9, count)
        unlocks[idx] = np.random.poisson(95, count)
        notifications[idx] = np.random.poisson(140, count)
        social_sessions[idx] = np.random.poisson(35, count)
        sleep_hours[idx] = np.random.normal(5.2, 1.1, count)
        physical_activity[idx] = np.random.normal(20, 10, count)
        outdoor_hours[idx] = np.random.normal(0.6, 0.3, count)

    elif p == 'Balanced User':
        comp_hours[idx] = np.random.normal(4.0, 1.0, count)
        smart_hours[idx] = np.random.normal(2.2, 0.6, count)
        tab_hours[idx] = np.random.normal(0.5, 0.3, count)
        social_hours[idx] = np.random.normal(1.1, 0.4, count)
        stream_hours[idx] = np.random.normal(1.2, 0.5, count)
        game_hours[idx] = np.random.normal(0.4, 0.3, count)
        prod_hours[idx] = np.random.normal(3.5, 0.8, count)
        night_hours[idx] = np.random.normal(0.3, 0.2, count)
        unlocks[idx] = np.random.poisson(35, count)
        notifications[idx] = np.random.poisson(55, count)
        social_sessions[idx] = np.random.poisson(12, count)
        sleep_hours[idx] = np.random.normal(7.8, 0.5, count)
        physical_activity[idx] = np.random.normal(55, 15, count)
        outdoor_hours[idx] = np.random.normal(2.2, 0.6, count)

    elif p == 'Entertainment Heavy':
        comp_hours[idx] = np.random.normal(3.0, 1.2, count)
        smart_hours[idx] = np.random.normal(4.0, 1.0, count)
        tab_hours[idx] = np.random.normal(2.0, 0.8, count)
        social_hours[idx] = np.random.normal(2.0, 0.7, count)
        stream_hours[idx] = np.random.normal(4.8, 1.2, count)
        game_hours[idx] = np.random.normal(4.5, 1.4, count)
        prod_hours[idx] = np.random.normal(1.0, 0.4, count)
        night_hours[idx] = np.random.normal(2.8, 0.8, count)
        unlocks[idx] = np.random.poisson(75, count)
        notifications[idx] = np.random.poisson(90, count)
        social_sessions[idx] = np.random.poisson(22, count)
        sleep_hours[idx] = np.random.normal(6.1, 0.9, count)
        physical_activity[idx] = np.random.normal(18, 8, count)
        outdoor_hours[idx] = np.random.normal(0.7, 0.3, count)

# Clip physical bounds
comp_hours = np.clip(comp_hours, 0, 14)
smart_hours = np.clip(smart_hours, 0.2, 14)
tab_hours = np.clip(tab_hours, 0, 10)
social_hours = np.clip(social_hours, 0, 12)
stream_hours = np.clip(stream_hours, 0, 12)
game_hours = np.clip(game_hours, 0, 12)
prod_hours = np.clip(prod_hours, 0, 12)
night_hours = np.clip(night_hours, 0, 8)
sleep_hours = np.clip(sleep_hours, 3, 11)
physical_activity = np.clip(physical_activity, 0, 180)
outdoor_hours = np.clip(outdoor_hours, 0, 8)

daily_screen_time = smart_hours + comp_hours + tab_hours
avg_session_mins = np.clip((smart_hours * 60) / np.maximum(unlocks, 1), 1, 120)

first_screen_hour = np.clip(np.random.normal(7.0, 1.2, N) - (night_hours * 0.3), 4, 11)
last_screen_hour = np.clip(np.random.normal(22.5, 1.0, N) + (night_hours * 0.5), 20, 28)

def format_time(float_hours):
    h = np.floor(float_hours).astype(int) % 24
    m = np.floor((float_hours % 1) * 60).astype(int)
    return [f"{h_i:02d}:{m_i:02d}" for h_i, m_i in zip(h, m)]

first_screen_time = format_time(first_screen_hour)
last_screen_time = format_time(last_screen_hour)

social_interaction = np.clip(np.random.normal(2.5, 1.0, N) + (outdoor_hours * 0.3) - (game_hours * 0.2), 0, 10)
work_hours = np.clip(comp_hours * 0.8 + prod_hours * 0.5 + np.random.normal(1.0, 0.5, N), 0, 14)
break_freq = np.clip(np.random.poisson(5, N) + (10 - work_hours).clip(0, 10) * 0.5, 1, 20).astype(int)
weekend_screen_time = daily_screen_time * np.where(is_weekend, 1.15, 0.9)

# Scores with realistic stochastic variation (breaks 1:1 multicollinearity)
focus_score = np.clip(100 - (notifications * 0.18 + unlocks * 0.12 + (night_hours * 3.5)) + np.random.normal(0, 8.0, N), 10, 100)
productivity_score = np.clip((prod_hours * 8) + (focus_score * 0.25) - (social_hours * 2) + np.random.normal(0, 10.0, N), 0, 100)
digital_dependency = np.clip((daily_screen_time * 3) + (unlocks * 0.2) + (night_hours * 4) + np.random.normal(0, 9.0, N), 0, 100)
social_engagement = np.clip((social_interaction * 7) + (social_hours * 4) + np.random.normal(0, 10.0, N), 0, 100)
routine_consistency = np.clip(100 - (np.abs(sleep_hours - 7.5) * 8 + night_hours * 5) + np.random.normal(0, 8.0, N), 0, 100)
distraction_freq = np.clip(np.random.poisson(12, N) + (notifications * 0.1) + np.random.normal(0, 4.0, N), 0, 80).astype(int)

# -----------------------------------------------------------------------------
# 3. NON-LINEAR TARGET FORMULATION WITH NOISE
# -----------------------------------------------------------------------------
# Base balanced baseline
base = 58.0

# Non-linear terms & diminishing returns
sleep_effect = np.where(sleep_hours < 6.5, (sleep_hours - 6.5) * 4.5, (sleep_hours - 6.5) * 1.5)
activity_effect = np.log1p(physical_activity) * 3.2
outdoor_effect = np.sqrt(outdoor_hours) * 4.0
routine_effect = (routine_consistency - 50) * 0.20

night_penalty = (night_hours ** 1.3) * 3.0
dependency_penalty = (digital_dependency / 100.0) ** 1.5 * 18.0
distraction_penalty = np.sqrt(distraction_freq) * 2.2

# Complex multi-variable interactions
synergy_boost = np.where((physical_activity > 40) & (sleep_hours >= 7.0) & (night_hours < 1.0), 8.0, 0.0)
compound_strain = np.where((daily_screen_time > 9.0) & (night_hours > 2.5) & (sleep_hours < 6.0), -12.0, 0.0)

# Natural human variation (Gaussian noise)
noise = np.random.normal(0, 11.5, size=N)

raw_wellbeing = (
    base 
    + sleep_effect 
    + activity_effect 
    + outdoor_effect 
    + routine_effect 
    - night_penalty 
    - dependency_penalty 
    - distraction_penalty 
    + synergy_boost 
    + compound_strain 
    + noise
)

wellbeing_score = np.clip(raw_wellbeing, 0, 100).round(2)

# Balanced, realistic quantile-based category distribution
def assign_category(score):
    if score >= 68:
        return 'Excellent'
    elif score >= 54:
        return 'Good'
    elif score >= 40:
        return 'Moderate'
    else:
        return 'Poor'

wellbeing_category = np.vectorize(assign_category)(wellbeing_score)

# -----------------------------------------------------------------------------
# 4. EXPORT
# -----------------------------------------------------------------------------
df = pd.DataFrame({
    'user_id': user_ids,
    'age': ages,
    'gender': genders,
    'occupation': occupations,
    'work_schedule': work_schedules,
    'education_level': education_levels,
    'daily_screen_time_hours': daily_screen_time.round(2),
    'smartphone_usage_hours': smart_hours.round(2),
    'computer_usage_hours': comp_hours.round(2),
    'tablet_usage_hours': tab_hours.round(2),
    'social_media_hours': social_hours.round(2),
    'video_streaming_hours': stream_hours.round(2),
    'gaming_hours': game_hours.round(2),
    'productive_app_hours': prod_hours.round(2),
    'daily_notifications': notifications,
    'phone_unlocks': unlocks,
    'social_media_sessions': social_sessions,
    'average_session_minutes': avg_session_mins.round(2),
    'night_time_usage_hours': night_hours.round(2),
    'first_screen_time': first_screen_time,
    'last_screen_time': last_screen_time,
    'sleep_hours': sleep_hours.round(2),
    'physical_activity_minutes': physical_activity.round(1),
    'outdoor_time_hours': outdoor_hours.round(2),
    'social_interaction_hours': social_interaction.round(2),
    'work_hours': work_hours.round(2),
    'break_frequency': break_freq,
    'weekend_screen_time': weekend_screen_time.round(2),
    'focus_score': focus_score.round(1),
    'productivity_score': productivity_score.round(1),
    'digital_dependency_score': digital_dependency.round(1),
    'social_engagement_score': social_engagement.round(1),
    'routine_consistency_score': routine_consistency.round(1),
    'distraction_frequency': distraction_freq,
    'weekday': weekdays,
    'weekend': is_weekend,
    'remote_work': remote_work,
    'exam_period': exam_period,
    'holiday_period': holiday_period,
    'wellbeing_score': wellbeing_score,
    'wellbeing_category': wellbeing_category
})

file_name = "human_digital_behavior_wellbeing.csv"
df.to_csv(file_name, index=False)
print(f"Successfully generated '{file_name}' ({df.shape[0]:,} rows × {df.shape[1]} columns).")
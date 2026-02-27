"""
Precompute user features for all users in the dataset.
Output: users_features.json - used by backend for fast lookup.
"""
import pandas as pd
import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS_PATH = os.path.join(BASE_DIR, 'event_data_train.csv')
SUBMISSIONS_PATH = os.path.join(BASE_DIR, 'submissions_data_train.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'data', 'users_features.json')

# Базовые 13 признаков + 6 полиномиальных (как в лучшей XGB модели из notebook)
BASE_FEATURES = [
    'days', 'steps_tried', 'correct', 'wrong', 'correct_ratio', 'viewed', 'passed',
    'view_to_pass_ratio', 'first_try_ratio', 'active_hours', 'last_sub_correct',
    'attempts_per_step', 'first_day_ratio'
]
# Отобранные полиномиальные признаки (XGB Best ROC-AUC)
SELECTED_POLY_FEATURES = [
    'view_to_pass_ratio active_hours',  # view_to_pass_ratio * active_hours
    'days first_try_ratio',             # days * first_try_ratio
    'wrong viewed',                     # wrong * viewed
    'days wrong',                       # days * wrong
    'wrong^2',                         # wrong**2
    'steps_tried viewed'                # steps_tried * viewed
]
FEATURE_COLUMNS = BASE_FEATURES + SELECTED_POLY_FEATURES


def main():
    print("Precomputing user features...")
    events_data = pd.read_csv(EVENTS_PATH)
    submissions_data = pd.read_csv(SUBMISSIONS_PATH)

    events_data['date'] = pd.to_datetime(events_data.timestamp, unit='s')
    events_data['day'] = events_data.date.dt.date
    submissions_data['date'] = pd.to_datetime(submissions_data.timestamp, unit='s')
    submissions_data['day'] = submissions_data.date.dt.date

    # user_min_time
    user_min_time = events_data.groupby('user_id', as_index=False).agg(
        {'timestamp': 'min'}).rename(columns={'timestamp': 'min_timestamp'})
    three_days_sec = 3 * 24 * 60 * 60
    user_min_time['max_timestamp'] = user_min_time['min_timestamp'] + three_days_sec

    # Filter events to first 3 days
    events_merged = events_data.merge(user_min_time[['user_id', 'min_timestamp', 'max_timestamp']], on='user_id')
    events_data_train = events_merged[
        events_merged['timestamp'] <= events_merged['max_timestamp']
    ][['step_id', 'timestamp', 'action', 'user_id', 'date', 'day']]

    # Filter submissions to first 3 days
    sub_merged = submissions_data.merge(user_min_time[['user_id', 'min_timestamp', 'max_timestamp']], on='user_id')
    submissions_data_train = sub_merged[
        sub_merged['timestamp'] <= sub_merged['max_timestamp']
    ][['step_id', 'timestamp', 'submission_status', 'user_id', 'date', 'day']]

    # Compute features per user
    all_user_ids = set(events_data.user_id.unique()) | set(submissions_data.user_id.unique())

    X = submissions_data_train.groupby('user_id').day.nunique().to_frame().reset_index()
    X = X.rename(columns={'day': 'days'})

    steps_tried = submissions_data_train.groupby('user_id').step_id.nunique().to_frame().reset_index()
    steps_tried = steps_tried.rename(columns={'step_id': 'steps_tried'})
    X = X.merge(steps_tried, on='user_id', how='outer')

    sub_pivot = submissions_data_train.pivot_table(
        index="user_id", columns="submission_status", values="step_id",
        aggfunc="count", fill_value=0
    ).reset_index()
    if 'correct' in sub_pivot.columns and 'wrong' in sub_pivot.columns:
        X = X.merge(sub_pivot[['user_id', 'correct', 'wrong']], on='user_id', how='outer')
    else:
        X['correct'] = 0
        X['wrong'] = 0
    X = X.fillna(0)
    X['correct_ratio'] = X['correct'] / (X['correct'] + X['wrong'] + 1e-10)

    events_pivot = events_data_train.pivot_table(
        index="user_id", columns="action", values="step_id",
        aggfunc="count", fill_value=0
    ).reset_index()
    if 'viewed' in events_pivot.columns:
        X = X.merge(events_pivot[['user_id', 'viewed']], on='user_id', how='left')
    else:
        X['viewed'] = 0
    X['viewed'] = X['viewed'].fillna(0)

    if 'passed' in events_pivot.columns:
        X = X.merge(events_pivot[['user_id', 'passed']], on='user_id', how='left')
    else:
        X['passed'] = 0
    X['passed'] = X['passed'].fillna(0)

    X['view_to_pass_ratio'] = X['passed'] / (X['viewed'] + 1)

    first_attempts = submissions_data_train.sort_values('timestamp').drop_duplicates(
        subset=['user_id', 'step_id'], keep='first'
    )
    first_try_correct = first_attempts[first_attempts['submission_status'] == 'correct'].groupby(
        'user_id'
    ).size().reset_index(name='first_try_correct')
    X = X.merge(first_try_correct, on='user_id', how='left')
    X['first_try_correct'] = X['first_try_correct'].fillna(0)
    X['first_try_ratio'] = X['first_try_correct'] / (X['steps_tried'] + 1)
    X = X.drop('first_try_correct', axis=1, errors='ignore')

    time_features = events_data_train.groupby('user_id')['timestamp'].agg(['min', 'max']).reset_index()
    time_features['active_hours'] = (time_features['max'] - time_features['min']) / 3600
    X = X.merge(time_features[['user_id', 'active_hours']], on='user_id', how='left')
    X['active_hours'] = X['active_hours'].fillna(0)

    last_sub_idx = submissions_data_train.groupby('user_id')['timestamp'].idxmax()
    last_sub = submissions_data_train.loc[last_sub_idx][['user_id', 'submission_status']]
    last_sub['last_sub_correct'] = (last_sub['submission_status'] == 'correct').astype(int)
    X = X.merge(last_sub[['user_id', 'last_sub_correct']], on='user_id', how='left')
    X['last_sub_correct'] = X['last_sub_correct'].fillna(0)

    X['attempts_per_step'] = (X['correct'] + X['wrong']) / (X['steps_tried'] + 1)

    events_with_min = events_data_train.merge(
        time_features[['user_id', 'min']].rename(columns={'min': 'first_ts'}), on='user_id'
    )
    first_day_events = events_with_min[
        events_with_min['timestamp'] < events_with_min['first_ts'] + 86400
    ].groupby('user_id').size().reset_index(name='first_day_events')
    total_events = events_data_train.groupby('user_id').size().reset_index(name='total_events')
    X = X.merge(first_day_events, on='user_id', how='left')
    X = X.merge(total_events, on='user_id', how='left')
    X['first_day_events'] = X['first_day_events'].fillna(0)
    X['total_events'] = X['total_events'].fillna(1)
    X['first_day_ratio'] = X['first_day_events'] / X['total_events']
    X = X.drop(['first_day_events', 'total_events'], axis=1, errors='ignore')

    X = X[['user_id'] + BASE_FEATURES]

    # Добавляем полиномиальные признаки (как в notebook)
    X['view_to_pass_ratio active_hours'] = X['view_to_pass_ratio'] * X['active_hours']
    X['days first_try_ratio'] = X['days'] * X['first_try_ratio']
    X['wrong viewed'] = X['wrong'] * X['viewed']
    X['days wrong'] = X['days'] * X['wrong']
    X['wrong^2'] = X['wrong'] ** 2
    X['steps_tried viewed'] = X['steps_tried'] * X['viewed']

    X = X[['user_id'] + FEATURE_COLUMNS]

    # Build dict: user_id -> { features }
    result = {}
    for _, row in X.iterrows():
        uid = int(row['user_id'])
        result[uid] = {k: float(row[k]) for k in FEATURE_COLUMNS}

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print(f"Saved features for {len(result)} users to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()

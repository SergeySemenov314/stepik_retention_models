"""
Stepik Retention Model - Training script
Обучает лучшую XGBoost модель (XGB Best ROC-AUC) с полиномиальными признаками.
Сохраняет model.pkl, feature_config.pkl, poly.pkl для инференса.
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS_PATH = os.path.join(BASE_DIR, 'event_data_train.csv')
SUBMISSIONS_PATH = os.path.join(BASE_DIR, 'submissions_data_train.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_service', 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_FEATURES = [
    'days', 'steps_tried', 'correct', 'wrong', 'correct_ratio', 'viewed', 'passed',
    'view_to_pass_ratio', 'first_try_ratio', 'active_hours', 'last_sub_correct',
    'attempts_per_step', 'first_day_ratio'
]
SELECTED_POLY = [
    'view_to_pass_ratio active_hours', 'days first_try_ratio', 'wrong viewed',
    'days wrong', 'wrong^2', 'steps_tried viewed'
]
# Лучшие параметры из notebook (XGB Best ROC-AUC)
XGB_PARAMS = {
    'subsample': 0.9, 'reg_lambda': 1, 'reg_alpha': 1, 'n_estimators': 400,
    'min_child_weight': 1, 'max_depth': 3, 'learning_rate': 0.01,
    'gamma': 0, 'colsample_bytree': 0.9
}


def load_and_prepare_data():
    """Load events and submissions, prepare users_data with min_timestamp."""
    print("Loading data...")
    events_data = pd.read_csv(EVENTS_PATH)
    submissions_data = pd.read_csv(SUBMISSIONS_PATH)

    events_data['date'] = pd.to_datetime(events_data.timestamp, unit='s')
    events_data['day'] = events_data.date.dt.date
    submissions_data['date'] = pd.to_datetime(submissions_data.timestamp, unit='s')
    submissions_data['day'] = submissions_data.date.dt.date

    # users_data with last_timestamp, is_gone_user, passed_course
    users_data = events_data.groupby('user_id', as_index=False).agg(
        {'timestamp': 'max'}).rename(columns={'timestamp': 'last_timestamp'})
    now = events_data.timestamp.max()
    drop_out_threshold = 30 * 24 * 60 * 60  # 30 days
    users_data['is_gone_user'] = (now - users_data.last_timestamp) > drop_out_threshold

    # users_score from submissions
    users_score = submissions_data.pivot_table(
        index="user_id", columns="submission_status", values="step_id",
        aggfunc="count", fill_value=0
    ).reset_index()
    users_data = users_data.merge(users_score, on='user_id', how='outer')
    users_data = users_data.fillna(0)

    # users_events_data
    users_events_data = events_data.pivot_table(
        index="user_id", columns="action", values="step_id",
        aggfunc="count", fill_value=0
    ).reset_index()
    users_data = users_data.merge(users_events_data, on='user_id', how='outer')
    users_data = users_data.fillna(0)

    # users_days
    users_days = events_data.groupby('user_id').day.nunique().to_frame().reset_index()
    users_days = users_days.rename(columns={'day': 'day'})
    users_data = users_data.merge(users_days, on='user_id', how='outer')
    users_data = users_data.fillna(0)

    users_data['passed_course'] = users_data.passed > 170

    # user_min_time
    user_min_time = events_data.groupby('user_id', as_index=False).agg(
        {'timestamp': 'min'}).rename(columns={'timestamp': 'min_timestamp'})
    users_data = users_data.merge(user_min_time, on='user_id', how='outer')

    # Filter events and submissions to first 3 days
    three_days_sec = 3 * 24 * 60 * 60
    events_with_min = events_data.merge(user_min_time, on='user_id', how='left')
    events_with_min['max_timestamp'] = events_with_min['min_timestamp'] + three_days_sec
    events_data_train = events_with_min[
        events_with_min['timestamp'] <= events_with_min['max_timestamp']
    ].drop(columns=['min_timestamp', 'max_timestamp'], errors='ignore')

    sub_merged = submissions_data.merge(
        users_data[['user_id', 'min_timestamp']], on='user_id', how='left'
    )
    mask = (sub_merged['min_timestamp'].notna() &
            (sub_merged['timestamp'] <= sub_merged['min_timestamp'] + three_days_sec))
    submissions_data_train = submissions_data[mask].copy()

    return events_data_train, submissions_data_train, users_data


def compute_features(events_data_train, submissions_data_train, users_data):
    """Compute feature matrix X and target y."""
    # Base features from submissions
    X = submissions_data_train.groupby('user_id').day.nunique().to_frame().reset_index()
    X = X.rename(columns={'day': 'days'})

    steps_tried = submissions_data_train.groupby('user_id').step_id.nunique().to_frame().reset_index()
    steps_tried = steps_tried.rename(columns={'step_id': 'steps_tried'})
    X = X.merge(steps_tried, on='user_id', how='outer')

    sub_pivot = submissions_data_train.pivot_table(
        index="user_id", columns="submission_status", values="step_id",
        aggfunc="count", fill_value=0
    ).reset_index()
    X = X.merge(sub_pivot[['user_id', 'correct', 'wrong']], on='user_id', how='outer')
    X = X.fillna(0)
    X['correct_ratio'] = X['correct'] / (X['correct'] + X['wrong'] + 1e-10)

    # viewed from events
    events_pivot = events_data_train.pivot_table(
        index="user_id", columns="action", values="step_id",
        aggfunc="count", fill_value=0
    ).reset_index()
    X = X.merge(events_pivot[['user_id', 'viewed']], on='user_id', how='left')
    X['viewed'] = X['viewed'].fillna(0)

    # passed
    X = X.merge(events_pivot[['user_id', 'passed']], on='user_id', how='left')
    X['passed'] = X['passed'].fillna(0)

    X['view_to_pass_ratio'] = X['passed'] / (X['viewed'] + 1)

    # first_try_ratio
    first_attempts = submissions_data_train.sort_values('timestamp').drop_duplicates(
        subset=['user_id', 'step_id'], keep='first'
    )
    first_try_correct = first_attempts[first_attempts['submission_status'] == 'correct'].groupby(
        'user_id'
    ).size().reset_index(name='first_try_correct')
    X = X.merge(first_try_correct, on='user_id', how='left')
    X['first_try_correct'] = X['first_try_correct'].fillna(0)
    X['first_try_ratio'] = X['first_try_correct'] / (X['steps_tried'] + 1)
    X = X.drop('first_try_correct', axis=1)

    # active_hours
    time_features = events_data_train.groupby('user_id')['timestamp'].agg(['min', 'max']).reset_index()
    time_features['active_hours'] = (time_features['max'] - time_features['min']) / 3600
    X = X.merge(time_features[['user_id', 'active_hours']], on='user_id', how='left')
    X['active_hours'] = X['active_hours'].fillna(0)

    # last_sub_correct
    last_sub_idx = submissions_data_train.groupby('user_id')['timestamp'].idxmax()
    last_sub = submissions_data_train.loc[last_sub_idx][['user_id', 'submission_status']]
    last_sub['last_sub_correct'] = (last_sub['submission_status'] == 'correct').astype(int)
    X = X.merge(last_sub[['user_id', 'last_sub_correct']], on='user_id', how='left')
    X['last_sub_correct'] = X['last_sub_correct'].fillna(0)

    X['attempts_per_step'] = (X['correct'] + X['wrong']) / (X['steps_tried'] + 1)

    # first_day_ratio
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
    X = X.drop(['first_day_events', 'total_events'], axis=1)

    # Merge with users_data for target
    X = X.merge(users_data[['user_id', 'passed_course', 'is_gone_user']], how='outer')

    # Filter: exclude users who are still active (not gone) and didn't pass
    X = X[~((X.is_gone_user == False) & (X.passed_course == False))]

    y = X.passed_course.map(int)
    X = X.drop(['passed_course', 'is_gone_user'], axis=1)
    X = X[BASE_FEATURES].fillna(0)

    return X, y


def main():
    print("Stepik Retention Model - Training")
    print("=" * 50)

    events_data_train, submissions_data_train, users_data = load_and_prepare_data()
    print(f"Events (first 3 days): {len(events_data_train)} rows")
    print(f"Submissions (first 3 days): {len(submissions_data_train)} rows")

    X, y = compute_features(events_data_train, submissions_data_train, users_data)
    print(f"Training samples: {len(X)}")
    print(f"Class balance: {y.value_counts().to_dict()}")

    # Полиномиальные признаки (как в notebook)
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X)
    poly_names = poly.get_feature_names_out(BASE_FEATURES)
    X_poly_df = pd.DataFrame(X_poly, columns=poly_names, index=X.index)
    X_final = pd.concat([X, X_poly_df[SELECTED_POLY]], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1,
        eval_metric='logloss', **XGB_PARAMS
    )
    model.fit(X_train, y_train)

    from sklearn.metrics import accuracy_score, roc_auc_score
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    joblib.dump(model, os.path.join(OUTPUT_DIR, 'model.pkl'))
    joblib.dump({'base_features': BASE_FEATURES, 'selected_poly_features': SELECTED_POLY},
                os.path.join(OUTPUT_DIR, 'feature_config.pkl'))
    joblib.dump(poly, os.path.join(OUTPUT_DIR, 'poly.pkl'))
    print(f"\nМодель сохранена в {OUTPUT_DIR}")


if __name__ == '__main__':
    main()

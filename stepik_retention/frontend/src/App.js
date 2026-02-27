import React, { useState, useEffect } from 'react';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || '';

const INTEGER_FEATURES = new Set([
  'days', 'steps_tried', 'correct', 'wrong', 'viewed', 'passed',
  'last_sub_correct', 'wrong^2', 'wrong viewed', 'days wrong', 'steps_tried viewed'
]);

const FEATURE_LABELS = {
  days: 'Дней активности',
  steps_tried: 'Шагов с попытками',
  correct: 'Правильных ответов',
  wrong: 'Неправильных ответов',
  correct_ratio: 'Доля правильных',
  viewed: 'Просмотрено шагов',
  passed: 'Пройдено шагов',
  view_to_pass_ratio: 'Конверсия просмотр→прохождение',
  first_try_ratio: 'Доля с первой попытки',
  active_hours: 'Часов активности',
  last_sub_correct: 'Последний ответ правильный',
  attempts_per_step: 'Попыток на шаг',
  first_day_ratio: 'Доля активности в 1-й день',
  'view_to_pass_ratio active_hours': 'view_to_pass × active_hours',
  'days first_try_ratio': 'days × first_try_ratio',
  'wrong viewed': 'wrong × viewed',
  'days wrong': 'days × wrong',
  'wrong^2': 'wrong²',
  'steps_tried viewed': 'steps_tried × viewed'
};

function App() {
  const [userIds, setUserIds] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingUsers, setLoadingUsers] = useState(true);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/users`);
        if (!res.ok) throw new Error('Failed to load users');
        const data = await res.json();
        setUserIds(data.userIds || []);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoadingUsers(false);
      }
    };
    fetchUsers();
  }, []);

  const pickRandomUser = async () => {
    if (userIds.length === 0) return;
    setError(null);
    setResult(null);
    const randomId = userIds[Math.floor(Math.random() * userIds.length)];
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_URL}/api/predict/${randomId}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.message || data.error || 'Request failed');
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="header">
        <h1>Stepik Retention Model</h1>
        <p className="subtitle">Предсказание прохождения онлайн курса пользователем по его активности за 3 дня</p>
      </header>

      <main className="main">
        {loadingUsers ? (
          <div className="loading-state">Загрузка списка пользователей...</div>
        ) : (
          <>
            <div className="action-block">
              <div className="action-description">
                <p className="action-description-main">
                  Нажмите кнопку — модель выберет случайного пользователя и предскажет,
                  пройдёт ли он курс до конца
                </p>
                <p className="action-description-sub">
                  {userIds.length.toLocaleString('ru-RU')} пользователей курса{' '}
                  <a
                    href="https://stepik.org/course/129/syllabus"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    «Анализ данных в R»
                  </a>{' '}
                  на Stepik
                </p>
              </div>

              <button
                className="random-btn"
                onClick={pickRandomUser}
                disabled={loading || userIds.length === 0}
              >
                {loading ? '⏳ Загрузка...' : '🎲 Случайный пользователь'}
              </button>
            </div>

            {error && (
              <div className="error-box">
                {error}
              </div>
            )}

            {result && (
              <div className="result-card">
                <div className="user-id-header">
                  <h2>Пользователь #{result.userId}</h2>
                </div>

                <div className="model-output">
                  <div className="model-output-label">Прогноз модели</div>
                  <div className="model-output-content">
                    <div className={`prediction-badge ${result.willComplete ? 'complete' : 'incomplete'}`}>
                      {result.prediction}
                    </div>
                    <div className="probability">
                      Вероятность прохождения: {(result.probability * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div className="user-data-section">
                  <h3>Данные о пользователе за первые 3 дня</h3>
                  <div className="features-grid">
                    {Object.entries(result.userData || {}).map(([key, value]) => (
                      <div key={key} className="feature-item">
                        <span className="feature-label">{FEATURE_LABELS[key] || key}</span>
                        <span className="feature-value">
                          {typeof value === 'number'
                            ? INTEGER_FEATURES.has(key)
                              ? value.toFixed(0)
                              : value < 1 && value > 0
                              ? value.toFixed(3)
                              : value.toFixed(1)
                            : value}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

export default App;

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

const ML_DESCRIPTION = [
  {
    title: 'Исходные данные',
    text: <>На старте были только сырые логи действий пользователей курса <a href="https://stepik.org/course/129/syllabus" target="_blank" rel="noopener noreferrer">«Анализ данных в R»</a> на Stepik за 3 года: таблица событий (просмотр, прохождение, начало попытки шага) и таблица сабмитов (правильные / неправильные ответы) с unix-таймстемпами. Никаких готовых признаков — только сырые события.</>
  },
  {
    title: 'Целевая переменная',
    text: 'Пользователь считается прошедшим курс, если он выполнил более 170 шагов.'
  },
  {
    title: 'Предобработка данных',
    text: 'Для каждого пользователя было найдено время первого события и отфильтрованы только действия в рамках первых 3 дней активности. Это имитирует реальный сценарий: предсказываем удержание на раннем этапе, не зная будущего.'
  },
  {
    title: 'Дисбаланс классов',
    text: 'В процессе анализа стало понятно, что задача имеет сильный дисбаланс классов: из ~17 985 пользователей курс прошли только ~1 425 (около 8%). Accuracy стала бессмысленной метрикой — модель, всегда предсказывающая "не пройдёт", даёт 92% точности. В качестве основной метрики был выбран ROC-AUC, так как он оценивает качество ранжирования пользователей по вероятности прохождения курса независимо от порога классификации.'
  },
  {
    title: 'Фиче инжиниринг',
    text: 'Из событий было извлечено 13 базовых признаков: количество дней активности, шагов с попытками, правильных и неправильных ответов, доля правильных с первой попытки, конверсия просмотр→прохождение, часы активности, последний результат и др. Затем методом перебора были протестированы 91 полиномиальный признак (квадраты и попарные произведения). Лучшие 6 добавлены в итоговый набор: view_to_pass × active_hours, days × first_try_ratio, wrong × viewed, days × wrong, wrong², steps_tried × viewed.'
  },
  {
    title: 'Выбор модели',
    text: 'Было обучено и сравнено 4 модели с учётом дисбаланса классов (class_weight / scale_pos_weight): Logistic Regression, Random Forest, Gradient Boosting и XGBoost. XGBoost показал лучший ROC-AUC. Для него был проведён RandomizedSearchCV по 100 конфигурациям гиперпараметров с оптимизацией по ROC-AUC.'
  },
  {
    title: 'Итоговая модель',
    text: 'Выбрана XGBoost с ROC-AUC = 0.8413. Из пользователей, реально прошедших курс, модель верно определяет 73.3%. Из тех, кто курс бросил — верно определяет ~79.7%.'
  },
  {
    title: 'Деплой',
    text: 'Модель упакована в Docker-контейнер с FastAPI-сервисом инференса. Рядом развёрнут Node.js бэкенд, который хранит предвычисленные признаки пользователей и проксирует запросы к модели. Фронтенд на React обращается к бэкенду.'
  }
];

function App() {
  const [userIds, setUserIds] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingUsers, setLoadingUsers] = useState(true);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showModal, setShowModal] = useState(false);

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
      <button className="info-btn" onClick={() => setShowModal(true)}>
        Как модель создавалась?
      </button>

      <header className="header">
        <h1>Stepik Retention Model</h1>
        <p className="subtitle">Предсказание прохождения онлайн курса пользователем по его активности за 3 дня</p>
      </header>

      {showModal && (
        <div className="modal-overlay" onClick={() => setShowModal(false)}>
          <div className="modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Как создавалась модель</h2>
              <button className="modal-close" onClick={() => setShowModal(false)}>✕</button>
            </div>
            <div className="modal-body">
              {ML_DESCRIPTION.map((section, i) => (
                <div key={i} className="modal-section">
                  <div className="modal-section-number">{i + 1}</div>
                  <div className="modal-section-content">
                    <h3>{section.title}</h3>
                    <p>{section.text}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

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

<<<<<<< HEAD
# reg_analysis_for_kaggle
(RU) VSCode project featuring voting ensemble making predictions on real-estate prices for House Prices - Advanced Regression Techniques competition (Kaggle)
=======
# Проект: Прогнозирование цен на недвижимость

## Описание проекта

Данный проект направлен на прогнозирование цен на недвижимость с использованием различных методов машинного обучения и библиотек для анализа данных. В проекте используются такие модели, как XGBoost, LightGBM и HistGradientBoostingRegressor, а также ансамблевый метод Voting Regressor для улучшения точности прогноза. Визуализация данных осуществляется с помощью Plotly и Dash.

## Структура проекта

- `proj_main.py` - основной скрипт проекта, который содержит:
  - Импорт необходимых библиотек (таких как `shap`, `optuna`, `dash`, `XGBRegressor`, `LGBMRegressor`, и другие).
  - Функции для анализа мультиколлинеарности, визуализации данных, оптимизации гиперпараметров моделей и другие.
  - Реализацию пользовательского интерфейса с помощью Dash для отображения результатов анализа и визуализаций.
  - Одиночное использование plotly и matplotlib для отображения графиков распределения данных, корреляций и SHAP-анализа.

## Используемые библиотеки

- `numpy`, `pandas` - для работы с данными
- `plotly`, `matplotlib` - для визуализации данных
- `optuna` - для оптимизации гиперпараметров моделей
- `xgboost`, `lightgbm`, `sklearn` - для построения моделей машинного обучения
- `dash` - для создания веб-интерфейса для взаимодействия с пользователем
- `shap` - для оценки важности признаков

## Основные функции

1. **Анализ данных и предобработка**: Используются `pandas` и `numpy` для загрузки и обработки данных. Пропуски в данных заполняются на основе логических правил, для оценки мультиколлинеарности используется коэффициент инфляции дисперсии (VIF). Также производится создание новых признаков и преобразование категориальных признаков с использованием различных видов кодировки.

2. **Моделирование**: Обучение моделей с использованием XGBRegressor, LGBMRegressor и HistGradientBoostingRegressor, а также их объединение в ансамбль с использованием Voting Regressor для повышения точности прогноза. Для оценки важности признаков используется библиотека `shap`.

3. **Оптимизация гиперпараметров**: Использование `optuna` для поиска оптимальных гиперпараметров моделей. Применяется `Hyperband Pruner` для эффективного выбора параметров и ускорения процесса оптимизации.

4. **Визуализация**: Визуализация данных и результатов моделирования с использованием `plotly` и `matplotlib`. Построение тепловой карты корреляции, визуализация важности признаков и графиков распределения данных.

5. **Веб-интерфейс**: В начале проекта предусмотрена реализация веб-интерфейса с использованием Dash для интерактивного взаимодействия с результатами анализа и визуализаций. Пользователь может исследовать важность признаков, распределение данных и результаты моделей через удобный интерфейс.

6. **Создание сабмишена**: Реализация функции для создания итогового сабмишена на основе Voting Regressor. Предсказания моделей обратно логарифмируются и сохраняются в формате CSV для отправки на платформу Kaggle.

## Запуск проекта

1. Клонируйте репозиторий:

   ```sh
   git clone <https://github.com/kumitayy/reg_analysis_for_kaggle>
   ```

2. Установите виртуальное окружение venv:
   ```sh
   python -m venv venv
   .\venv\Scripts\activate
   ```
   
3. Установите необходимые зависимости:

   ```sh
   pip install -r requirements.txt
   ```

4. Запустите основной скрипт:

   ```sh
   python proj_main.py
   ```

## Примечание
Проект выполнялся с использованием среды разработки VSCode.

## Использование

После запуска проекта, вы сможете получить доступ к веб-интерфейсу, который позволит вам визуализировать данные, оценивать результаты моделей и анализировать важность признаков.

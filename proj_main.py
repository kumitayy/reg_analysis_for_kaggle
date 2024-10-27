import logging
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import dash
from dash import dcc, html
from lightgbm import LGBMRegressor
from optuna.pruners import HyperbandPruner
from plotly.subplots import make_subplots
from sklearn.ensemble import VotingRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, ShuffleSplit
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from threading import Thread
from xgboost import XGBRegressor

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Загрузка данных
data_train = pd.read_csv('train.csv').drop(columns='Id').copy(deep=True)
data_test = pd.read_csv('test.csv').drop(columns='Id').copy(deep=True)

# Настройки отображения Pandas
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# Анализ пропущенных значений
logging.info('Анализ пропущенных значений')
missing_values_train = data_train.columns[data_train.isnull().any()].tolist()
missing_values_test = data_test.columns[data_test.isnull().any()].tolist()
logging.info(f'Столбцы с пропущенными значениями:\nTrain: {missing_values_train}\nTest: {missing_values_test}')
# -------------------------------------------------------------------------------------------------------------------------------


def visualize_data(df):
    """
    Визуализация данных с использованием Dash (распределение SalePrice, boxplot для числовых признаков и др.)
    """
    app = dash.Dash(__name__)

    # Гистограмма для SalePrice
    fig_hist = px.histogram(df, x='SalePrice', nbins=30, title='Histogram of SalePrice', template='plotly_dark')
    fig_hist.update_layout(height=1000, width=1870)

    # Boxplot для числовых признаков
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    fig_box = px.box(df, y=numeric_cols, title='Boxplot for Continuous Features', template='plotly_dark')
    fig_box.update_layout(height=1000, width=1870)

    # Корреляционная матрица
    corr_matrix = df.select_dtypes(include=np.number).corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", title='Correlation Matrix', template='plotly_dark')
    fig_corr.update_layout(height=2000, width=1850)

    # Barplot для категориальных признаков
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    fig_bar_combined = make_subplots(rows=len(categorical_cols), cols=1, subplot_titles=categorical_cols, vertical_spacing=0.01)

    for i, col in enumerate(categorical_cols):
        bar_fig = px.bar(df, x=col).data[0]
        fig_bar_combined.add_trace(bar_fig, row=i + 1, col=1)

    fig_bar_combined.update_layout(height=17500, width=1870, title_text="Combined Barplots", showlegend=False, template='plotly_dark')

    # Макет для Dash
    app.layout = html.Div([
        html.H1("Visualization Dashboard"),
        dcc.Tabs([
            dcc.Tab(label='SalePrice Histogram', children=[dcc.Graph(figure=fig_hist)]),
            dcc.Tab(label='Boxplot', children=[dcc.Graph(figure=fig_box)]),
            dcc.Tab(label='Correlation Matrix', children=[dcc.Graph(figure=fig_corr)]),
            dcc.Tab(label='Combined Barplots', children=[dcc.Graph(figure=fig_bar_combined)]),
        ]) 
    ])

    # Запуск сервера Dash и логирование URL
    thread = Thread(target=lambda: app.run_server(debug=True, use_reloader=False))
    thread.start()

visualize_data(data_train)
# -------------------------------------------------------------------------------------------------------------------------------


def clean_missing_columns(data_train, data_test, threshold=20):
    """
    Очистка столбцов с пропущенными значениями на основе порога и отображение DataFrame с пропусками.
    """
    # Объединение данных для анализа пропущенных значений
    combined_df = pd.concat([data_train, data_test], keys=['Train', 'Test'], axis=0)
    
    # Подсчет общего количества и процента пропущенных значений
    total_miss = combined_df.isnull().sum()
    percent_miss = (combined_df.isnull().sum() / combined_df.shape[0]) * 100
    
    # Создание DataFrame с пропущенными значениями
    miss_df = pd.DataFrame({'Общее количество пропусков': total_miss, 'Процент пропусков': percent_miss})
    miss_df = miss_df.sort_values(by='Общее количество пропусков', ascending=False)
    
    # Логирование сводки пропущенных данных
    logging.info(f'Сводка пропущенных значений (Топ-20):\n{miss_df.head(20)}')
    
    # Получение списка столбцов для удаления на основе порога
    columns_to_drop = miss_df[miss_df['Процент пропусков'] > threshold].index.tolist()

    # Убедиться, что 'SalePrice' не удаляется из тренировочного набора данных
    if 'SalePrice' in columns_to_drop:
        columns_to_drop.remove('SalePrice')
    
    # Удаление столбцов из тренировочных и тестовых данных
    data_train_cleaned = data_train.drop(columns=columns_to_drop)
    data_test_cleaned = data_test.drop(columns=[col for col in columns_to_drop if col in data_test.columns])

    logging.info(f'Удаленные столбцы: {columns_to_drop}')
    
    return data_train_cleaned, data_test_cleaned, miss_df

data_train_cleaned, data_test_cleaned, miss_df = clean_missing_columns(data_train, data_test)
# -------------------------------------------------------------------------------------------------------------------------------


def fill_missing_values(data_train, data_test):
    """
    Заполняет пропущенные значения в тренировочных и тестовых наборах данных
    с учетом специфических правил, используя np.nan, средние или модальные значения.
    """
    data_to_fill = [data_train, data_test]

    def fill_numeric(df, col, dependent_col=None):
        """
        Функция для заполнения пропущенных числовых данных.
        """
        if dependent_col:
            df.loc[df[dependent_col].isnull(), col] = 0
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    def fill_categorical(df, col, dependent_col=None):
        """
        Функция для заполнения пропущенных категориальных данных.
        """
        if dependent_col:
            df.loc[df[dependent_col].isnull(), col] = np.nan
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)

    for df in data_to_fill:
        # Обработка числовых столбцов
        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                if col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtUnfSF', 'TotalBsmtSF']:
                    fill_numeric(df, col, dependent_col='TotalBsmtSF')
                elif col in ['GarageArea', 'GarageCars']:
                    fill_numeric(df, col, dependent_col='GarageArea')
                elif col == 'GarageYrBlt':
                    fill_numeric(df, col, dependent_col='GarageArea')
                else:  # Для других числовых столбцов
                    fill_numeric(df, col)

        # Обработка категориальных столбцов
        categorical_cols = df.select_dtypes(include='object').columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                if col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
                    fill_categorical(df, col, dependent_col='BsmtQual')
                elif col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
                    fill_categorical(df, col, dependent_col='GarageQual')
                else:  # Для других категориальных столбцов
                    fill_categorical(df, col)

    return data_train, data_test

data_train_filled, data_test_filled = fill_missing_values(data_train_cleaned, data_test_cleaned)

# Проверка на NaN после заполнения пропусков
remaining_nan_columns_train = [col for col in data_train_filled.columns if data_train_filled[col].isnull().any()]
remaining_nan_columns_test = [col for col in data_test_filled.columns if data_test_filled[col].isnull().any()]

if remaining_nan_columns_train:
    logging.info(f'Столбцы data_train с NaN значениями после заполнения NaN: {remaining_nan_columns_train}')
else:
    logging.info('В тренировочных данных не осталось NaN.')

if remaining_nan_columns_test:
    logging.info(f'Столбцы data_test с NaN значениями после заполнения NaN: {remaining_nan_columns_test}')
else:
    logging.info('В тестовых данных не осталось NaN.')
# -------------------------------------------------------------------------------------------------------------------------------


def calculate_vif(df):
    """
    Вычисляет VIF для каждого признака в DataFrame и возвращает DataFrame
    с результатами.

    Параметры:
    df (DataFrame): Датасет с числовыми признаками.

    Возвращает:
    DataFrame: Признаки и их соответствующие значения VIF.
    """
    # Удаляем столбец 'SalePrice' если он присутствует
    if 'SalePrice' in df.columns:
        df = df.drop('SalePrice', axis=1)

    # Создаем DataFrame для хранения результатов VIF
    vif_data = pd.DataFrame()
    vif_data['Feature'] = df.columns
    vif_data['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]

    return vif_data

vif_results = calculate_vif(data_train_filled.select_dtypes(include=[np.number])).sort_values(by='VIF', ascending=False)
high_vif_results = vif_results[vif_results['VIF'] > 5]
logging.info(f'Отображение результатов теста на мультиколлинеарность:\n{high_vif_results}')
# -------------------------------------------------------------------------------------------------------------------------------


def plot_correlation(data_train, x_cols, y_col='SalePrice', plot_width=750, plot_height=400):
    """
    Строит scatter-плоты корреляции нескольких признаков с целевой переменной SalePrice.
    
    Параметры:
    data_train (DataFrame): Тренировочный набор данных.
    x_cols (list): Список имен признаков для оси X.
    y_col (str): Имя целевой переменной (по умолчанию 'SalePrice').
    plot_width (int): Ширина каждого графика.
    plot_height (int): Высота каждого графика.
    """
    # Проверка, является ли x_cols списком, если нет, преобразовать в список
    if isinstance(x_cols, str):
        x_cols = [x_cols]

    num_plots = len(x_cols)
    
    if num_plots == 1:
        # Если один столбец, то строим обычный график
        fig = px.scatter(data_train, x=x_cols[0], y=y_col, 
                            title=f'Корреляция {x_cols[0]} с {y_col}',
                            labels={x_cols[0]: x_cols[0], y_col: y_col},
                            trendline="ols", template='plotly_dark')
        fig.update_layout(width=plot_width, height=plot_height)  # Устанавливаем размер графика
    else:
        # Если больше одного столбца, создаем subplots
        rows = (num_plots + 1) // 2  # Определяем количество строк (по 2 графика на строку)
        fig = make_subplots(rows=rows, cols=2, subplot_titles=x_cols,
                            vertical_spacing=0.1, horizontal_spacing=0.1)
        
        for i, x_col in enumerate(x_cols):
            logging.info(f'Построение графика корреляции свойства {x_col}')
            scatter = px.scatter(data_train, x=x_col, y=y_col, 
                                    labels={x_col: x_col, y_col: y_col},
                                    trendline="ols").data[0]  # Получаем trace графика
            
            row = (i // 2) + 1
            col = (i % 2) + 1
            fig.add_trace(scatter, row=row, col=col)
        
        # Общие настройки для всех субплотов
        fig.update_layout(template='plotly_dark', 
                            title=f'Корреляция нескольких признаков с {y_col}',
                            width=2 * plot_width,  # Ширина холста с учетом двух графиков на строку
                            height=rows * plot_height)  # Высота холста в зависимости от числа строк

    fig.show()

plot_correlation(data_train_filled, ['BsmtUnfSF', 'TotalBsmtSF', 'GarageYrBlt', 
                                        'TotRmsAbvGrd', 'KitchenAbvGr', 'BedroomAbvGr', 
                                        'GarageCars', 'GarageArea'])
# -------------------------------------------------------------------------------------------------------------------------------


def feature_engineering(data_train_filled, data_test_filled):
    """
    Функция для выполнения feature engineering: создание новых признаков и удаление дублирующих.
    Добавлена оценка важности признаков через SHAP перед и после удаления признаков.
    
    Параметры:
    - data_train_filled (DataFrame): Тренировочный набор данных.
    - data_test_filled (DataFrame): Тестовый набор данных.
    
    Возвращает:
    - Обновленные тренировочные и тестовые данные.
    """
    data_to_clean = [data_train_filled, data_test_filled]
    
    for data in data_to_clean:
        # Создание новых признаков
        data['HouseAgeAtSale'] = data['YrSold'] - data['YearBuilt']
        data['RemodAgeAtSale'] = data['YrSold'] - data['YearRemodAdd']
        data['AvgRoomSize'] = data['GrLivArea'] / data['TotRmsAbvGrd']
        data['GarageScore'] = data['GarageArea'] * data['GarageCars']
        data['GarageAgeAtSale'] = data['YrSold'] - data['GarageYrBlt']
        data['TotalBaths'] = data['FullBath'] + data['HalfBath'] + data['BsmtFullBath'] + data['BsmtHalfBath']
        data['BathsRatio'] = (data['BsmtFullBath'] + data['BsmtHalfBath']) / (data['FullBath'] + data['HalfBath'] + 1e-5)
        data['TotalPorchSF'] = data['WoodDeckSF'] + data['OpenPorchSF'] + data['EnclosedPorch'] + data['3SsnPorch'] + data['ScreenPorch']

    # Оценка важности признаков до удаления с помощью SHAP
    X = data_train_filled.select_dtypes(include=['int64', 'float64']).drop(columns='SalePrice')
    y = data_train_filled['SalePrice']

    # Обучение XGBoost для оценки важности признаков через SHAP
    xgb_model = XGBRegressor()
    xgb_model.fit(X, y)

    # Используем SHAP для оценки важности признаков
    explainer = shap.Explainer(xgb_model, X)
    shap_values = explainer(X)

    # Визуализация SHAP-важности до удаления признаков
    shap.summary_plot(shap_values, X)
    
    logging.info('SHAP-оценка важности признаков перед удалением завершена.')

    # Удаление дублирующих признаков
    for data in data_to_clean:
        data.drop(columns=['BsmtFinSF1', 'BsmtFinSF2',  # Убираем в пользу TotalBsmtSF
                            '1stFlrSF', '2ndFlrSF',      # Убираем в пользу GrLivArea
                            'YrSold', 'YearBuilt', 'YearRemodAdd', 'TotRmsAbvGrd', 'ScreenPorch', 
                            'EnclosedPorch', '3SsnPorch', 'WoodDeckSF', 'OpenPorchSF', 
                            'GarageYrBlt', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 
                            'GarageCars', 'GarageArea'], inplace=True)

    # Оценка важности признаков после удаления с помощью SHAP
    X = data_train_filled.select_dtypes(include=['int64', 'float64']).drop(columns='SalePrice')
    xgb_model.fit(X, y)

    # Повторная визуализация SHAP-оценки важности признаков после удаления
    explainer = shap.Explainer(xgb_model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X)
    
    logging.info('SHAP-оценка важности признаков после удаления завершена.')

    # Пересчитываем VIF для числовых признаков в тренировочном наборе данных
    vif_results = calculate_vif(data_train_filled.select_dtypes(include=[np.number])).sort_values(by='VIF', ascending=False)
    logging.info(f'Мультиколлинеарность после feature engineering:\n{vif_results}')

    return data_train_filled, data_test_filled

data_train_refined, data_test_refined = feature_engineering(data_train_filled, data_test_filled)
# -------------------------------------------------------------------------------------------------------------------------------


def apply_ridge_regularization(data_train_refined, data_test_refined, alpha=1.0, unique_threshold=15):
    """
    Применяет Ridge-регрессию к числовым признакам с использованием RobustScaler,
    и логарифмирует целевую переменную для снижения влияния выбросов. Исключает признаки
    с малым числом уникальных значений (менее unique_threshold) из регуляризации.
    
    Параметры:
    - data_train_refined (DataFrame): Тренировочные данные после Feature Engineering.
    - data_test_refined (DataFrame): Тестовые данные после Feature Engineering (без SalePrice).
    - alpha (float): Коэффициент регуляризации для Ridge (по умолчанию 1.0).
    - unique_threshold (int): Порог уникальных значений для исключения признаков из регуляризации.
    
    Возвращает:
    - Обновленные тренировочные и тестовые данные с примененным scaling и регуляризацией.
    """
    # Применение логарифмирования к целевой переменной SalePrice в data_train_refined
    data_train_refined['SalePrice'] = np.log1p(data_train_refined['SalePrice'])  # Логарифмируем SalePrice

    # Определение числовых признаков для scaling и регуляризации
    numeric_columns = data_train_refined.select_dtypes(include=['int64', 'float64']).columns
    numeric_columns = numeric_columns.drop('SalePrice')  # Исключаем SalePrice из признаков

    # Исключение признаков, которые могут быть категориальными на основе уникальных значений
    numeric_columns = [col for col in numeric_columns if data_train_refined[col].nunique() >= unique_threshold]

    # Применяем RobustScaler для устойчивости к выбросам
    scaler = RobustScaler()

    # Масштабируем данные
    data_train_refined[numeric_columns] = scaler.fit_transform(data_train_refined[numeric_columns])
    data_test_refined[numeric_columns] = scaler.transform(data_test_refined[numeric_columns])

    # Применение Ridge-регрессии для уменьшения мультиколлинеарности
    ridge = Ridge(alpha=alpha)
    ridge.fit(data_train_refined[numeric_columns], data_train_refined['SalePrice'])

    # Преобразование признаков с использованием коэффициентов регуляризации
    data_train_refined[numeric_columns] = ridge.coef_ * data_train_refined[numeric_columns]
    data_test_refined[numeric_columns] = ridge.coef_ * data_test_refined[numeric_columns]

    # Возвращаем обновленные тренировочные и тестовые данные
    return data_train_refined, data_test_refined

data_train_refined, data_test_refined = apply_ridge_regularization(data_train_refined, data_test_refined, alpha=1.0, unique_threshold=15)
# -------------------------------------------------------------------------------------------------------------------------------


# Разделение столбцов на числовые и категориальные
numeric_columns = sorted(data_train_refined.select_dtypes(include=[np.number]).columns)
categorical_columns = sorted(data_train_refined.select_dtypes(include=['object', 'category']).columns)

logging.info(f'Числовые столбцы: {numeric_columns}')
logging.info(f'Категориальные столбцы: {categorical_columns}')

# Список категориальных столбцов с порядковыми значениями
ordinal_columns = ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'CentralAir', 'ExterCond', 
                    'ExterQual', 'Functional', 'GarageCond', 'GarageFinish', 'GarageQual', 'HeatingQC', 'KitchenQual', 
                    'PavedDrive', 'Street', 'Utilities']

# Распределение порядковых значений
ordinal_categories = {
    'BsmtCond': ['Po', 'Fa', 'TA', 'Gd'],  # от худшего к лучшему состоянию
    'BsmtExposure': ['No', 'Mn', 'Av', 'Gd'],  # от отсутствия окна до хорошего окна
    'BsmtFinType1': ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],  # от незавершенного до высокого качества
    'BsmtFinType2': ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],  # аналогично для второго типа
    'BsmtQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # от худшего к отличному качеству подвала
    'CentralAir': ['N', 'Y'],  # от отсутствия до наличия кондиционирования
    'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # от худшего к лучшему состоянию внешних конструкций
    'ExterQual': ['Fa', 'TA', 'Gd', 'Ex'],  # от худшего к лучшему качеству
    'Functional': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],  # от серьезных проблем до типичного состояния
    'GarageCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # от худшего к отличному состоянию гаража
    'GarageFinish': ['Unf', 'RFn', 'Fin'],  # от неотделанного до полностью отделанного
    'GarageQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # от худшего к лучшему качеству гаража
    'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],  # от худшего к лучшему качеству отопления
    'KitchenQual': ['Fa', 'TA', 'Gd', 'Ex'],  # от худшего к лучшему качеству кухни
    'PavedDrive': ['N', 'P', 'Y'],  # от неасфальтированного до полностью асфальтированного
    'Street': ['Grvl', 'Pave'],  # от гравийной до асфальтированной дороги
    'Utilities': ['ELO', 'NoSeWa', 'NoSewr', 'AllPub']  # от минимальных коммунальных услуг до всех общественных коммуникаций
}

def encode_data(train_df, test_df, ordinal_columns, ordinal_categories):
    """
    Преобразование категориальных столбцов с использованием различных подходов.
    
    - Ordinal Encoding для столбцов из списка ordinal_columns с заданными порядковыми категориями.
    - OneHotEncoding для столбцов с количеством уникальных значений < 10.
    - Frequency Encoding для столбцов с количеством уникальных значений >= 10.
    
    Параметры:
    train_df (DataFrame): Тренировочный датасет для кодирования.
    test_df (DataFrame): Тестовый датасет для кодирования.
    ordinal_columns (list): Список столбцов для Ordinal Encoding.
    ordinal_categories (dict): Словарь с порядковыми категориями для каждого столбца.
    
    Возвращает:
    DataFrame: Преобразованные тренировочные и тестовые данные.
    """
    logging.info('Начало процесса кодирования данных')

    # Отделение целевого признака SalePrice
    target = train_df['SalePrice']
    train_df = train_df.drop(columns=['SalePrice'])

    # Объединение тренировочных и тестовых данных для одинакового кодирования
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # Ordinal Encoding для столбцов с порядковыми категориями
    ordinal_encoder = OrdinalEncoder(categories=[ordinal_categories[col] for col in ordinal_columns])
    combined_df[ordinal_columns] = ordinal_encoder.fit_transform(combined_df[ordinal_columns])
    logging.info(f'Ordinal Encoding применен к столбцам: {ordinal_columns}')
    
    # Определение других категориальных столбцов для OneHotEncoding и FrequencyEncoding
    nominal_columns = combined_df.select_dtypes(include='object').columns.tolist()
    nominal_columns = [col for col in nominal_columns if col not in ordinal_columns]
    
    for col in nominal_columns:
        unique_values = combined_df[col].nunique()
        logging.info(f'Обрабатываем столбец {col} с {unique_values} уникальными значениями')

        if unique_values < 10:
            # OneHotEncoding для столбцов с количеством уникальных значений меньше 10
            combined_df = pd.get_dummies(combined_df, columns=[col], drop_first=True)
            logging.info(f'OneHotEncoding применен к столбцу: {col}')
        else:
            # Frequency Encoding для столбцов с количеством уникальных значений больше 10
            freq_encoding = combined_df[col].value_counts() / len(combined_df)
            combined_df[col] = combined_df[col].map(freq_encoding)
            logging.info(f'Frequency Encoding применен к столбцу: {col}')
    
    # Разделение обратно на тренировочные и тестовые наборы данных по количеству строк и копирование для избежания предупреждений
    train_df_encoded = combined_df.iloc[:len(train_df)].copy()
    test_df_encoded = combined_df.iloc[len(train_df):].copy()

    # Возвращаем целевой признак SalePrice в тренировочный набор
    train_df_encoded['SalePrice'] = target
    
    logging.info(f'Форма data_train_encoded: {train_df_encoded.shape}')
    logging.info(f'Форма data_test_encoded: {test_df_encoded.shape}')
    logging.info('Процесс кодирования данных завершен')
    
    return train_df_encoded, test_df_encoded

data_train_encoded, data_test_encoded = encode_data(data_train_refined, data_test_refined, ordinal_columns, ordinal_categories)
# -------------------------------------------------------------------------------------------------------------------------------


def plot_shap_feature_importance(data_train_encoded, target_column='SalePrice', model=XGBRegressor(), height=8000):
    """
    Строит интерактивный график важности признаков на основе SHAP значений после encoding.
    
    Параметры:
    - data_train_encoded (DataFrame): Датасет с закодированными признаками.
    - target_column (str): Имя столбца целевой переменной (по умолчанию 'SalePrice').
    - model (ML модель): Модель для обучения, по умолчанию XGBRegressor().
    - height (int): Высота графика для Plotly (по умолчанию 8000 для больших наборов признаков).
    
    Возвращает:
    - SHAP Summary Plot в Plotly.
    """
    
    # Разделение признаков и целевой переменной
    X = data_train_encoded.drop(columns=target_column)
    y = data_train_encoded[target_column]
    
    # Преобразование булевых колонок в int
    bool_columns = X.select_dtypes(include='bool').columns
    X[bool_columns] = X[bool_columns].astype(int)
    
    # Обучение модели
    model.fit(X, y)
    
    # Вычисление SHAP значений
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    # Преобразование SHAP значений в DataFrame
    shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
    
    # Рассчет средних абсолютных значений SHAP для каждого признака
    shap_importance = shap_df.abs().mean().sort_values(ascending=False)
    
    # Преобразуем в DataFrame для удобства использования с Plotly
    shap_summary_df = pd.DataFrame({
        'Feature': shap_importance.index,
        'Mean SHAP Value': shap_importance.values
    })
    
    # Строим интерактивный бар-чарт с Plotly
    fig = px.bar(
        shap_summary_df, 
        x='Mean SHAP Value', 
        y='Feature', 
        title='SHAP Summary Plot (Plotly)',
        height=height,  # Можно задать высоту под количество признаков
        template='plotly_dark'
    )
    
    # Показываем график
    fig.show()
    logging.info('Отображение SHAP после декодирования категориальных стобцов завершено')

plot_shap_feature_importance(data_train_encoded)
# -------------------------------------------------------------------------------------------------------------------------------


def perform_rfe_selection(X_train, y_train, X_test, n_features_to_select):
    """
    Применение Recursive Feature Elimination (RFE) с XGBoost для выбора наиболее важных признаков и выделение их из тестового набора.
    
    Параметры:
    - X_train (DataFrame): Набор признаков для обучения.
    - y_train (Series): Целевой признак.
    - X_test (DataFrame): Тестовый набор данных для выделения тех же признаков.
    - n_features_to_select (int): Количество признаков, которые нужно оставить.
    
    Возвращает:
    - X_train_selected (DataFrame): Тренировочный набор с отобранными признаками.
    - X_test_selected (DataFrame): Тестовый набор с отобранными признаками.
    - selected_features (list): Список выбранных признаков.
    """

    logging.info('Начало процесса оценивания важности признаков')

    # Инициализация модели XGBoost
    xgb_model = XGBRegressor(n_estimators=1000, random_state=42)
    
    # Инициализация RFE
    rfe = RFE(estimator=xgb_model, n_features_to_select=n_features_to_select, step=1)
    
    # Применение RFE
    rfe.fit(X_train, y_train)
    
    # Получение выбранных признаков
    selected_features = X_train.columns[rfe.support_].tolist()
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]  # Выделяем те же признаки из тестового набора
    
    logging.info(f"Выбранные признаки: {selected_features}")
    logging.info('Выбор признаков на основе важности завершен')

    return X_train_selected, X_test_selected, selected_features

X_train = data_train_encoded.drop(columns=['SalePrice'])
y_train = data_train_encoded['SalePrice']
X_test = data_test_encoded

# Оставляем 80 лучших признаков
X_train_selected, X_test_selected, selected_features = perform_rfe_selection(X_train, y_train, X_test, n_features_to_select=80)
# -------------------------------------------------------------------------------------------------------------------------------


# Функция для оценки модели
def evaluate_model(X, y, model):
    """
    Оценивает производительность модели с использованием RepeatedKFold и возвращает средние значения метрик.
    
    Параметры:
    - X (DataFrame): Признаки.
    - y (Series): Целевой признак.
    - model: Модель или ансамбль моделей для обучения.
    
    Возвращает:
    - rmse (float): Среднее значение RMSE.
    - r2 (float): Среднее значение R².
    """
    cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    rmse_scores = []
    r2_scores = []
    
    for train_ix, test_ix in cv.split(X):
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        
        # Обучение модели или ансамбля
        model.fit(X_train, y_train)
        
        # Прогнозирование
        y_pred = model.predict(X_test)
        
        # Вычисление метрик
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        rmse_scores.append(rmse)
        r2_scores.append(r2)
    
    return np.mean(rmse_scores), np.mean(r2_scores)

# Инициализация базовых моделей
lgbm_model = LGBMRegressor(random_state=42, verbose=-1)
xgb_model = XGBRegressor(random_state=42)
hgb_model = HistGradientBoostingRegressor(random_state=42)

# Инициализация VotingRegressor с добавлением GradientBoostingRegressor
voting_model = VotingRegressor(
    estimators=[
        ('lgbm', lgbm_model),
        ('xgb', xgb_model),
        ('hgb', hgb_model)
    ],
    n_jobs=-1
)

# Оценка модели ДО выбора признаков
X_train_full = data_train_encoded.drop(columns=['SalePrice'])
y_train = data_train_encoded['SalePrice']

rmse_full, r2_full = evaluate_model(X_train_full, y_train, voting_model)
logging.info(f"До выбора признаков - RMSE: {rmse_full}, R²: {r2_full}")

# Оценка модели ПОСЛЕ выбора признаков
X_train_selected = data_train_encoded[selected_features]

rmse_selected, r2_selected = evaluate_model(X_train_selected, y_train, voting_model)
logging.info(f"После выбора признаков - RMSE: {rmse_selected}, R²: {r2_selected}")
# -------------------------------------------------------------------------------------------------------------------------------


def optimize_with_optuna(X_train, y_train, model_name, n_trials=200):
    """
    Оптимизирует гиперпараметры для XGBRegressor, LGBMRegressor и HistGradientBoostingRegressor с использованием Optuna.
    
    Параметры:
    - X_train (DataFrame): Набор признаков для обучения.
    - y_train (Series): Целевой признак.
    - model_name (str): Название модели ('xgb', 'lgbm', 'hgb').
    - n_trials (int): Количество попыток для поиска гиперпараметров Optuna.
    
    Возвращает:
    - best_model: Модель с оптимальными гиперпараметрами.
    - best_params: Лучшие найденные гиперпараметры.
    - final_rmse: Финальное значение RMSE.
    - final_r2: Финальное значение R².
    """
    def objective(trial):
        # Определение гиперпараметров для XGBRegressor
        if model_name == 'xgb':
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'eta': trial.suggest_float('eta', 0.001, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 1e-8, 1e-2),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
            model = XGBRegressor(**param, random_state=42, n_jobs=-1)

        # Определение гиперпараметров для LGBMRegressor
        elif model_name == 'lgbm':
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 3000),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 30, 70),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
            model = LGBMRegressor(**param, random_state=42, n_jobs=-1)

        # Определение гиперпараметров для HistGradientBoostingRegressor
        elif model_name == 'hgb':
            param = {
                'max_iter': trial.suggest_int('max_iter', 100, 3000),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
                'l2_regularization': trial.suggest_float('l2_regularization', 0.0, 1.0),
                'max_bins': trial.suggest_int('max_bins', 128, 255)
            }
            model = HistGradientBoostingRegressor(**param, random_state=42)

        # Кросс-валидация с использованием ShuffleSplit
        cv = ShuffleSplit(n_splits=6, test_size=0.2, random_state=42)
        rmse_scores = []
        
        for train_ix, test_ix in cv.split(X_train):
            X_train_fold, X_test_fold = X_train.iloc[train_ix], X_train.iloc[test_ix]
            y_train_fold, y_test_fold = y_train.iloc[train_ix], y_train.iloc[test_ix]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_test_fold)
            rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    logging.info(f'Лучшие гиперпараметры для {model_name}: {best_params}')

    if model_name == 'xgb':
        best_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    elif model_name == 'lgbm':
        best_model = LGBMRegressor(**best_params, random_state=42, n_jobs=-1)
    elif model_name == 'hgb':
        best_model = HistGradientBoostingRegressor(**best_params, random_state=42)

    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_train)
    final_rmse = np.sqrt(mean_squared_error(y_train, y_pred_best))
    final_r2 = r2_score(y_train, y_pred_best)

    return best_model, best_params, final_rmse, final_r2

# Оптимизация и создание VotingRegressor
xgb_model, xgb_params, xgb_rmse, xgb_r2 = optimize_with_optuna(X_train_selected, y_train, model_name='xgb')
lgbm_model, lgbm_params, lgbm_rmse, lgbm_r2 = optimize_with_optuna(X_train_selected, y_train, model_name='lgbm')
hgb_model, hgb_params, hgb_rmse, hgb_r2 = optimize_with_optuna(X_train_selected, y_train, model_name='hgb')

# Ансамбль VotingRegressor с оптимизированными моделями
voting_model = VotingRegressor(
    estimators=[
        ('lgbm', LGBMRegressor(**lgbm_params, random_state=42)),
        ('xgb', XGBRegressor(**xgb_params, random_state=42)),
        ('hgb', HistGradientBoostingRegressor(**hgb_params, random_state=42))
    ]
)

rmse_voting, r2_voting = evaluate_model(X_train_selected, y_train, voting_model)
logging.info(f"Voting Ensemble - RMSE: {rmse_voting}, R²: {r2_voting}")
# -------------------------------------------------------------------------------------------------------------------------------


def create_ensemble_submission(models, X_train_selected, y_train, X_test, file_name='submission.csv'):
    """
    Создает сабмишен на основе VotingRegressor, проводит обратное логарифмирование предсказаний и сохраняет их в CSV файл.
    
    Параметры:
    - models (list): Список базовых моделей с оптимальными гиперпараметрами.
    - X_train_selected (pd.DataFrame): Набор признаков обучающих данных.
    - y_train (pd.Series): Целевой признак обучающих данных.
    - X_test (pd.DataFrame): Набор признаков тестовых данных.
    - file_name (str): Имя файла для сохранения (по умолчанию 'submission.csv').
    
    Возвращает:
    - submission (pd.DataFrame): DataFrame с двумя столбцами: 'Id' и 'SalePrice'.
    """
    
    # Создание VotingRegressor с базовыми моделями
    ensemble_model = VotingRegressor(
        estimators=models,
        n_jobs=-1
    )
    
    # Обучение ансамбля на обучающих данных
    ensemble_model.fit(X_train_selected, y_train)
    
    # Получение предсказаний на тестовых данных
    y_pred_ensemble = ensemble_model.predict(X_test)
    
    # Применение обратного логарифмирования
    y_pred_exp = np.expm1(y_pred_ensemble)  # Используем expm1 для обратного log1p
    
    # Создание DataFrame для submission
    submission = pd.DataFrame({
        'Id': range(1461, 1461 + len(y_pred_exp)),  # Идентификаторы начинаются с 1461
        'SalePrice': y_pred_exp
    })
    
    # Сохранение в CSV файл
    submission.to_csv(file_name, index=False)
    
    print(f'Submission файл сохранен как {file_name}')
    
    return submission

# Оптимизированные модели
models = [
    ('lgbm', LGBMRegressor(**lgbm_params, random_state=42, verbose=-1)),
    ('xgb', XGBRegressor(**xgb_params, random_state=42)),
    ('hgb', HistGradientBoostingRegressor(**hgb_params, random_state=42))
]

# Создание submission.csv
create_ensemble_submission(models, X_train_selected, y_train, X_test_selected, file_name='submission.csv')
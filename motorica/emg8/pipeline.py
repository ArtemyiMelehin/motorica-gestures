# Работа с табличными данными
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy.stats as stats

# Преобразование признаков
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Пайплайны
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Модели, валидация, метрики
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score

# Подбор гиперпараметров
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

import os

# Для аннотаций
from typing import List, Any

from motorica.emg8.constants import *

from motorica.emg8.markers import *


# ----------------------------------------------------------------------------------------------
# ПАРАМЕТРЫ ЧТЕНИЯ ИСХОДНЫХ ДАННЫХ И РАЗМЕТКИ

# DATA_DIR = 'data/new'

# N_OMG_CH = 16         # количество каналов OMG-датчиков
# OMG_COL_PRFX = 'omg'  # префикс в названиях столбцов датафрейма, соответствующих OMG-датчикам
# STATE_COL = 'state'   # столбец с названием жеста, соответствующего команде
# CMD_COL = 'id'        # столбец с меткой команды на выполнение жеста
# TS_COL = 'ts'         # столбец метки времени
# N_GESTS_IN_CYCLE = 14 # количество жестов в цикле протокола (включая нейтральные)

# NOGO_STATE = 'Neutral'      # статус, обозначающий нейтральный жест
# BASELINE_STATE = 'Baseline' # доп. служебный статус в начале монтажа
# FINISH_STATE   = 'Finish'   # доп. служебный статус в конце монтажа

# # Список с названиями всех столбцов OMG
# OMG_CH = [OMG_COL_PRFX + str(i) for i in range(N_OMG_CH)]

# # Новые столбцы:
# SYNC_COL = 'sample'   # порядковый номер размеченного жеста
# GROUP_COL = 'group'   # новый группы (цикла протокола)
# TARGET = 'act_label'  # таргет (метка фактически выполняемого жеста)



# ----------------------------------------------------------------------------------------------
# ПОДГРУЗКА И РАЗМЕТКА
        
def read_emg8(
        montage: str, 
        dir: str = DATA_DIR, 
        sep: str = ' ',
        feature_cols: List[str] = OMG_CH,
        cmd_col: str = CMD_COL,
        state_col: str = STATE_COL,
        ts_col: str = TS_COL,
        sync_col_name: str = SYNC_COL,
        group_col_name: str = GROUP_COL,
        states_to_drop: list = [BASELINE_STATE, FINISH_STATE],
        marker = BasePeakMarker(),
        target_col_name: str = TARGET,
        n_holdout_groups: int = 0
        ) -> List:
    """
    Осуществляет чтение файла с данными измерений монтажа .emg8.

    Перенумеровывает классы жестов в порядке их следования по протоколу монтажа.

    Добавляет в возвращаемый датафрейм признак `sync_col_name`, представляющий собой порядковый номер жеста в монтаже.

    Находит циклы протокола и маркирует их признаком `group_col_name`
    ### Параметры

    montage : str
        имя файла для чтения

    dir : str, default="data"
        название папки, в которой находится файл

    sep :  str, default=' '
        символ-разделитель в исходном файле

    feature_cols: List[str], default=OMG_COLS
        список названий столбцов, которые будут использоваться в качестве признаков

    cmd_col : str, default=CMD_COL

    **ts_col**: *str, default=TS_COL*

    **sync_col_name**: *str, default=SYNC_COL*

    **drop_baseline_and_finish**: *bool, default=True*<br>
    Удалять ли в начале и в конце монтажа измерения с метками `Baseline` и `Finish` соответственно

    **mark_up**: *bool, default=True*

    **target_col_name**: *str, default=TARGET*

    **n_holdput_groups**: *int, default=0*

    ### Возвращаемый результат

    Список: **X_train**, **X_test** | None, **y_train**, **y_train** | None, **data**
    """

    path = os.path.join(dir, montage)

    data_origin = pd.read_csv(path, sep=sep, index_col=None)

    # Удалим примеры с указанными статусами 
    # (имеются ввиду служебные дополнительные статусы 'Baseline' и 'Finish')
    if not states_to_drop is None:
        mask = ~data_origin[state_col].isin(states_to_drop)
        data_origin = data_origin[mask]

    # Перенумеруем классы жестов в порядке их появления в монтаже
    states_ordered = data_origin[state_col].drop_duplicates().reset_index(drop=True)
    states_ordered = pd.Series(states_ordered.index, index=states_ordered)
    data_origin[cmd_col] = data_origin[state_col].apply(lambda state: states_ordered[state])

    # Далее работаем только с необходимыми столбцами
    data = data_origin.copy()[feature_cols + [cmd_col, state_col, ts_col]]
    
    # Добавим признак порядкового номера жеста (требуется для алгоритма разметки)
    bounds = data[data[cmd_col] != data[cmd_col].shift(1)].index
    bounds = np.append(bounds, data.index[-1])
    for i, lr in enumerate(zip(bounds, bounds[1:])):
        l, r = lr # l, r - индексы начала текущей и следующей эпох соответственно
        data.loc[l: r, sync_col_name] = i
    data[sync_col_name] = data[sync_col_name].astype(int)

    data = marker.fit_transform(data)

    # ПЕРЕПИШЕМ признак sync_col (порядковый номер жеста в монтаже), 
    # чтобы он соответствовал разметке ПО ФАКТИЧЕСКИМ ГРАНИЦАМ
    bounds = data[data[target_col_name] != data[target_col_name].shift(1)].index
    for i, lr in enumerate(zip(bounds, np.append(bounds[1:], [data.index[-1]]))):
        l, r = lr # l, r - индексы начала текущей и следующей эпох соответственно
        data.loc[l: r - 1, sync_col_name] = i

    # Определим кол-во жестов в одном цикле протокола
    protocol = data[[target_col_name, sync_col_name]].drop_duplicates()[target_col_name]
    #protocol = protocol[protocol != 0].to_numpy()
    for n in range(1, len(protocol)):
        sld = sliding_window_view(protocol, n)[::n]
        # Если все элементы (метки классов) равны 
        if (sld == sld[0]).all():
            n_gests_in_group = n #* 2 # Умножим на 2, чтобы учесть нейтральный жест
            break

    # Теперь проставим номера групп (циклов протокола) – понадобится при кросс-валидации
    group_idx = list(data[[target_col_name, sync_col_name]].drop_duplicates().index[::n_gests_in_group])
    # [последний индекс "дотянем" до конца]
    group_idx[-1] = data.index[-1] + 1
    for i, lr in enumerate(zip(group_idx, group_idx[1:])):
        l, r = lr
        data.loc[l: r - 1, group_col_name] = i
    n_groups = i

    # Разделяем признаки и таргет
    X = data[feature_cols].copy()
    y = data[target_col_name].copy()

    # Если задано количество циклов (групп) для отложенной выборки, 
    # разделяем данные на train и test
    if n_holdout_groups > 0:
        if n_holdout_groups >= n_groups:
            raise ValueError(f"Количество отложенных групп n_holdput_groups={n_holdout_groups} должно быть меньше общего количества групп {n_groups + 1}")
        last_train_idx = group_idx[-n_holdout_groups - 1] - 1
        X_train = X.loc[: last_train_idx, :].to_numpy()
        X_test = X.loc[last_train_idx + 1: , :].to_numpy()
        y_train = y.loc[: last_train_idx].to_numpy()
        y_test = y.loc[last_train_idx + 1: ].to_numpy()
        groups = data.loc[: last_train_idx, group_col_name].to_numpy()
    else: # n_holdout_groups == 0
        # Если не требуется выделять отложенную тестовую группу, 
        # то в train-выборку вернем все данные
        X_train = X.to_numpy()
        X_test = None 
        y_train = y.to_numpy()
        y_test = None
        groups = data[group_col_name].to_numpy()

    # Чтобы вернуть полные исходные данные, 
    # но с добавлением меток таргета, порядкового номера жеста и номера цикла протокола, 
    # просто скопируем соответствующие столбцы:
    data_origin[target_col_name] = data[target_col_name] # таргет
    data_origin[sync_col_name] = data[sync_col_name]     # порядковый номер жеста
    data_origin[group_col_name] = data[group_col_name]   # порядковый номер цикла

    return X_train, X_test, y_train, y_test, data_origin, groups


# ----------------------------------------------------------------------------------------------
# ПРЕОБРАЗОВАНИЕ И СОЗДАНИЕ ПРИЗНАКОВ

# Самый первый обработчик: переводит одномерный вектор признаков (когда нужно 
# предсказать один пример) в двумерный массив

class FixOneDimSample(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape((1, -1))
        return X

# Базовый класс для преобразования данных с помощью скользящего окна
class BaseSlidingProc(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            n_lags: int = 3,
            oper: Literal['add', 'replace'] = 'replace'
        ):
        self.n_lags = n_lags
        self.oper = oper

    # По умолчанию для всех дочерних классов считаем, 
    # что их работа не вносит задержку при работе в реальном времени.
    def get_realtime_shift(self):
        return 0

    def fit(self, X, y=None):
        self.n_ftr = X.shape[1]
        self.X_que = np.empty((0, self.n_ftr))
        return self
    

    # В дочерних классах данных внутренний метод 
    # должен реализовавыть фактическую обработку
    def _proc_func(self, X_sld):
        # В качестве заглушки просто возвращаем последний (текущий) пример
        return X_sld[:, -1]
    

    def transform(self, X):

        X = np.array(X)

        if self.n_lags < 2:
            return X
        
        # Если наш объект получает данные впервые,
        if self.X_que.shape[0] == 0:
            # накопируем первый пример нужное число раз
            self.X_que = np.tile(X[0], (self.n_lags - 1, 1))

        self.X_que = np.vstack((self.X_que, X))

        # Скользящее окно по двумерному массиву признаков 
        X_sld = sliding_window_view(self.X_que, (self.n_lags, self.n_ftr))[:, 0]

        X_proc = self._proc_func(X_sld)

        # В зависимости от значение oper 
        if self.oper == 'add':
            # добавляем новый признак к основному набору
            X_res = np.hstack((X, X_proc))
        else: # self.oper == 'replace'
            # либо заменяем исходные признаки новыми значениями
            X_res = X_proc

        # Запомним в нашем объекте лаги только для будущего нового примера
        self.X_que = self.X_que[-self.n_lags + 1: ]

        return X_res
    
    # Для преобразования обучающей выборки (fit_transform)
    # нам нельзя вносить задержку во временные ряды
    def fit_transform(self, X, y=None):
        self.fit(X)
        X_proc = self.transform(X)
        self.X_que = np.empty((0, self.n_ftr))
        shift = self.get_realtime_shift()
        if shift == 0:
            return X_proc
        return np.vstack((X_proc[shift:, :], X[-shift:, :]))


# Преобразователь для сглаживания ("срезания"  выбросов)
class NoiseReduction(BaseSlidingProc):

    def __init__(
            self,
            n_lags: int = 3,
            oper: Literal['add', 'replace'] = 'replace',
            avg: str = 'median' # 'mean'
        ):
        super().__init__(n_lags=n_lags, oper=oper)
        self.avg = avg

    def get_realtime_shift(self):
        return self.n_lags // 2

    def _proc_func(self, X_sld):
        
        np_avg_func = {
            'mean': np.mean,
            'median': np.median
            }[self.avg]
        X_avg = np_avg_func(X_sld, axis=1)
        
        return X_avg
    

# Обозачение тренда (нахождение разности с предыдущими значениями)
# class AddDifference(BaseSlidingProc):

#     def __init__(
#             self,
#             n_lags: int = 4,
#             avg: str = 'mean', # 'median'
#             n_features: int = N_OMG_CH
#         ):
#         super().__init__(n_lags=n_lags, oper='add')
#         self.avg = avg
#         self.n_features = n_features


#     def _proc_func(self, X_sld, X):
        
#         np_avg_func = {
#             'mean': np.mean,
#             'median': np.median
#             }[self.avg]
#         X_prev_avg = np_avg_func(X_sld[:, :-1, :self.n_features], axis=1)
#         # Результат – разность признаков и их средних предыдущих значений
#         return X[:, :self.n_features] - X_prev_avg
    
class AddDifference(BaseSlidingProc):

    def __init__(
            self,
            n_lags: int = 4,
            oper: Literal['add', 'replace'] = 'add',
            avg: str = 'mean', # 'median'
            # n_features: int = N_OMG_CH
        ):
        super().__init__(n_lags=n_lags, oper=oper)
        self.avg = avg
        # self.n_features = n_features


    def _proc_func(self, X_sld):
        
        np_avg_func = {
            'mean': np.mean,
            'median': np.median
            }[self.avg]
        # X_prev_avg = np_avg_func(X_sld[:, :-1, :self.n_features], axis=1)
        X_prev_avg = np_avg_func(X_sld[:, :-1], axis=1)
        # Результат – разность признаков и их средних предыдущих значений
        return X_sld[:, -1] - X_prev_avg
    


# class AvgDiff(BaseEstimator, TransformerMixin):

#     def __init__(self, scaler=None, avg='mean'):
#         self.scaler = scaler
#         self.avg = avg


#     def fit(self, X, y=None):
#         if not self.scaler is None:
#             self.scaler.fit(X)
#         return self
    
    
#     def transform(self, X):
#         if not self.scaler is None:
#             X = self.scaler.transform(X)
#         else:
#             X = np.array(X)
#         np_func = {
#             'mean': np.mean,
#             'median': np.median
#         }[self.avg]
#         X_avg = np_func(X, axis=1).reshape((-1, 1))
#         return np.hstack([X, X - X_avg])
    
class Gradients(BaseSlidingProc):

    def __init__(
            self,
            n_lags: int = 3,
            oper: Literal['add', 'replace'] = 'replace',
            spacing: int = 3
        ):
        super().__init__(n_lags=n_lags, oper=oper)
        self.spacing = spacing

    def _proc_func(self, X_sld):
        
        X_grad1 = np.gradient(X_sld, self.spacing, axis=1)
        # X_grad2 = np.gradient(X_grad1, 3, axis=1)
    
        # return np.hstack([
        #     X_grad1.reshape(X_sld.shape[0], -1), X_grad2.reshape(X_sld.shape[0], -1)
        # ])

        return X_grad1.reshape((X_sld.shape[0], -1))

    

# ----------------------------------------------------------------------------------------------
# ОБЕРТКА МОДЕЛИ С ПОСТОБРАБОТКОЙ ПРЕДСКАЗАНИЙ

class PostprocWrapper(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            estimator = LogisticRegression(C=10, max_iter=5000),
            n_lags: int = 5
        ):
        self.estimator = estimator
        self.n_lags = n_lags
        self.y_que = np.empty((0,))

        self.classes_ = []

    def get_realtime_shift(self):
        return self.n_lags - 1

    def fit(self, X, y):
        # Обучаем модель на лаговых признаках. 
        # При этом первые w - 1 примеров исходных данных пропускаются 
        # и в обучении не участвуют 
        self.estimator.fit(X, y)
        # Скопируем атрибут модели .classes_ в соответсвутующий атрибут обёртки
        try:
            self.classes_ = self.estimator.classes_
        except AttributeError:
            self.classes_ = pd.Series(y).unique().tolist()
        return self
    
    # В дочерних классах данных внутренний метод 
    # должен реализовавыть фактическую постобработку
    def _proc_func(self, yy):
        # В базовой версии возвращаем моду
        mode_res, _ = stats.mode(yy)
        return mode_res

    def predict(self, X):

        y_pred = self.estimator.predict(X)

        if self.n_lags < 2:
            return y_pred
        
        # Если наш объект-обёртка получает данные впервые,
        if self.y_que.shape[0] == 0:
            # накопируем первый ответ нужное кол-во раз
            self.y_que = np.tile(y_pred[0], self.n_lags - 1)

        self.y_que = np.hstack((self.y_que, y_pred))

        y_pred_proc = np.array(
            [self._proc_func(yy) for yy in sliding_window_view(self.y_que, self.n_lags)]
        )

        # Запомним в нашем объекте лаги только для будущего нового предсказания
        self.y_que = self.y_que[-self.n_lags + 1: ]

        return y_pred_proc
    
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    
    def set_params(self, **params):

        self.y_que = np.empty(0)
        # Все именованные аргументы (параметры) кроме перечисленных
        # предназначены для оборачиваемой модели
        wrapper_params = ['n_lags']

        params_for_estimator = dict()
        params_for_self = dict()
        for param in params:
            if param in wrapper_params:
                params_for_self[param] = params[param]
            else:
                params_for_estimator[param] = params[param]

        self.estimator.set_params(**params_for_estimator)
        return super().set_params(**params_for_self)
    

class TransWrapper(PostprocWrapper):
    # В дочерних классах данных внутренний метод 
    # должен реализовавыть фактическую постобработку
    def _proc_func(self, yy):
        yy = yy.copy()
        yy[yy < 0] = 0
        yy = yy % 10
        # В базовой версии возвращаем моду
        mode_res, _ = stats.mode(yy)
        return mode_res



# ----------------------------------------------------------------------------------------------
# ПАЙПЛАЙНЫ

MAX_TOTAL_SHIFT = 10

# Получить суммарный лаг пайплайна (в количестве измерений)
def get_total_shift(pipeline: Pipeline):
    total_shift = 0
    for i in range(len(pipeline)):
        try:
            total_shift += pipeline[i].get_realtime_shift()
        except AttributeError:
            continue
    return total_shift


# Пайплайн на базе логистической регрессии
def create_grad_logreg_pipeline( 
        X, y,
        exec_optimize: bool = False,
        groups=None,
        exec_fit: bool = True,
        max_total_shift: int = MAX_TOTAL_SHIFT,
        n_trials: int = 100
    ):

    pl = Pipeline([
        ('fix_1dim_sample', FixOneDimSample()),
        ('noise_reduct', NoiseReduction(3)),
        ('gradients', Gradients(3, 'add')),
        ('scaler', MinMaxScaler()),
        # ('model', PostprocWrapper(estimator=LogisticRegression(C=10, max_iter=5000)))
        ('model', TransWrapper(estimator=LogisticRegression(C=10, max_iter=5000), n_lags=7))
    ])

    def opt_func(trial: optuna.Trial, X=X, y=y, groups=groups, pl=pl, max_total_shift=max_total_shift):

        params = {
            'noise_reduct__n_lags': trial.suggest_int('noise_reduct__n_lags', 1, 5),
            'model__n_lags':        trial.suggest_int('model__n_lags', 3, 7, step=2),
            'gradients__n_lags':    trial.suggest_int('gradients__n_lags', 2, 5),
            'model__C':             trial.suggest_int('model__C', 1, 50, log=True)
        }
        pl.set_params(**params)

        total_shift = get_total_shift(pl)
    
        if total_shift == 0:
            y_shifted = y
        else:
            total_shift = min(total_shift, max_total_shift)
            y_shifted = np.hstack((np.tile(y[0], total_shift), y[: -total_shift]))

        n_groups = int(np.max(groups) + 1)
        kf = StratifiedGroupKFold(n_groups)
        scores = []
        for index_train, index_valid in kf.split(X, y, groups=groups):
            X_train = X[index_train]
            y_train = y_shifted[index_train]
            X_valid = X[index_valid]
            y_valid = y_shifted[index_valid]
            pl.fit(X_train, y_train)
            y_pred = pl.predict(X_valid)
            scores.append(f1_score(y_valid, y_pred, average='macro'))
        return np.mean(scores)
    
    if exec_optimize:
        study = optuna.create_study(direction='maximize')
        study.optimize(opt_func, n_trials=n_trials, show_progress_bar=True)
        pl.set_params(**study.best_params)

    if exec_fit:  
        pl.fit(X, y)
    
    return pl
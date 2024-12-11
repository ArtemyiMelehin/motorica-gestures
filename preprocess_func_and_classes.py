import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, LayerNormalization, Dropout


LABELS = {
  'Neutral': 0,
 'Finish': -1,
 'Close': 1,
 'Indication': 8,
 'Open': 2,
 'Pinch': 7,
 'ThumbFingers': 6,
 'Wrist_Extend': 4,
 'Wrist_Flex': 3,
 'Baseline': -1
 }

OMG_CHANELS_CNT = 16  # количество каналов OMG-датчиков
OMG_COL_PRFX = 'omg'  # префикс в названиях столбцов датафрейма, соответствующих OMG-датчикам
LABEL_COL = 'id'      # столбец таргета
TS_COL = 'ts'         # столбец метки времени

SYNC_COL = 'sample'   # порядковый номер размеченного жеста - для синхронизации и группировок
TARGET = 'act_label'  # таргет - метка фактически выполняемого жеста

# Сформируем список с названиями всех столбцов OMG
OMG_CH = [OMG_COL_PRFX + str(i) for i in range(OMG_CHANELS_CNT)]


# Класс для разметки данных
class BasePeakMarker(BaseEstimator, TransformerMixin):
    '''
    Класс-преобразователь для добавления в данные признака `"act_label"` – метки фактически выполняемого жеста.

    ### Параметры объекта класса

    **sync_col**: *str, default=SYNC_COL*<br>
    Название столбца с порядковым номером жеста (в соответствии с поступившей командой)

    **label_col**: *str, default=LABEL_COL*<br>
    Название столбца с меткой жеста (в соответствии с поступившей командой)

    **ts_col**: *str, default=TS_COL*<br>
    Название столбца с меткой времени

    **hi_val_threshold**: *float, default=0.1*<br>
    Нижний порог медианы показаний датчика (в нормализованных измерениях)
    для отнесения его к числу *активных* датчиков

    **sudden_drop_threshold**: *float, default=0.1*<br>
    Верхний порог отношения первого процентиля к медиане показаний датчика
    для отнесения его к числу датчиков, которым свойственны внезапные падения сигнала
    до околонулевых значений (такие датчики игнорируются при выполнении разметки)

    **sync_shift**: *int, default=0*<br>
    Общий сдвиг меток синхронизации (признак `sync_col`)

    **bounds_shift**: *int, default=0*<br>
    Корректировка (сдвиг) найденных границ

    **clean_w**: *int, default=5*<br>
    Параметр используемый при очистке найденных максимумов (пиков):
    если два или более пиков находятся на расстоянии не более `clean_w` измерений друг от друга,
    то из них оставляем один самый высокий пик

    **use_grad2**: *bool, default=True*<br>
    Если True - алгоритм разметки использует локальные максимумы (пики) суммарного второго градиента
    показаний датчиков. Иначе - используется градиент суммарного стандартного отклонения

    ## Методы
    Данный класс реализует стандартные методы классов-преобразователей *scikit-learn*:

    `fit()`, `fit_transform()` и `transform()` и может быть использован как элемент пайплайна.
    '''

    def __init__(
        self,
        sync_col: str = SYNC_COL,
        label_col: str = LABEL_COL,
        ts_col: str = TS_COL,
        hi_val_threshold: float = 0.1,
        sudden_drop_threshold: float = 0.1,
        sync_shift: int = 0,
        bounds_shift: int = 0,
        clean_w: int = 5,
        use_grad2: bool = True
    ):
        self.sync_col = sync_col
        self.label_col = label_col
        self.ts_col = ts_col
        self.hi_val_threshold = hi_val_threshold
        self.sudden_drop_threshold = sudden_drop_threshold
        self.sync_shift = sync_shift
        self.bounds_shift = bounds_shift
        self.clean_w = clean_w
        self.use_grad2 = use_grad2

    # Внутренний метод для нахождения "пиков" (второго градиента, градиента стандартного отклонения)
    def _find_peaks(
        self,
        X: np.ndarray,
        window: int,
        spacing: int,
    ):
        def _peaks(arr):
            mask = np.hstack([
                [False],
                (arr[: -2] < arr[1: -1]) & (arr[2:] < arr[1: -1]),
                [False]
            ])
            peaks = arr.copy()
            peaks[~mask] = 0
            peaks[peaks < 0] = 0
            return peaks

        def _clean_peaks(arr, w=self.clean_w):
            peaks = arr.copy()
            # Разберемся с пиками, расположенными близко друг к другу:
            # из нескольких пиков, помещающихся в окне w,
            # оставим только один - максимальный
            for i in range(peaks.shape[0] - w + 1):
                slice = peaks[i: i + w]
                max_peak = np.max(slice)
                mask = slice != max_peak
                peaks[i: i + w][mask] = 0
            return peaks

        # 1) Градиенты векторов показаний датчиков
        grad1 = np.sum(np.abs(np.gradient(X, spacing, axis=0)), axis=1)
        # не забудем заполнить образовавшиеся "дырки" из NaN
        grad1 = np.nan_to_num(grad1)

        grad2 = np.gradient(grad1, spacing, axis=0)
        grad2 = np.nan_to_num(grad2)
        peaks2 = _peaks(grad2)

        # 2) Среднее стандартное отклонение и его градиент
        std = np.mean(pd.DataFrame(X).rolling(window, center=True).std(), axis=1)
        std = np.nan_to_num(std)

        std1 = np.gradient(std, 1, axis=0)
        std1 = np.nan_to_num(std1)
        peaks_std1 = _peaks(std1)

        # Возвращаем пики градиента стандартного отклонения и второго градиента
        return _clean_peaks(peaks_std1), _clean_peaks(peaks2)

    # Функция для непосредственной разметки
    def _mark(
        self,
        X: pd.DataFrame
    ) -> np.ndarray[int]:

        # Сглаживание
        X_omg = pd.DataFrame(X[self.mark_sensors]).rolling(self.window, center=True).median()
        # Приведение к единому масштабу
        X_omg = MinMaxScaler((1, 1000)).fit_transform(X_omg)

        peaks_std1, peaks_grad2 = self._find_peaks(
            X_omg,
            window=self.window,
            spacing=self.spacing
        )

        peaks = peaks_grad2 if self.use_grad2 else peaks_std1

        sync = X[self.sync_col].copy()
        # Сдвигаем синхронизацию
        if self.sync_shift > 0:
            sync.iloc[self.sync_shift:] = sync.iloc[: -self.sync_shift]
            sync.iloc[: self.sync_shift] = 0

        # Искать максимальные пики будем внутри отрезков,
        # определяемых по признаку синхронизации
        sync_mask = sync != sync.shift(-1)
        sync_index = X[sync_mask].index

        labels = [int(X.loc[idx + 1, self.label_col]) for idx in sync_index[:-1]]

        bounds = np.array([])

        for l, r in zip(sync_index, sync_index[1:]):
            bounds = np.append(bounds, np.argmax(peaks[l: r]) + l)

        X_mrk = X.copy()
        X_mrk[TARGET] = 0

        for i, lr in enumerate(zip(bounds, np.append(bounds[1:], X_mrk.index[-1] + 1))):
            l, r = lr
            # l, r - индексы начала текущего и следующего жестов соответственно
            X_mrk.loc[l: r, TARGET] = labels[i]

        return X_mrk, bounds + self.bounds_shift

    def fit(self, X: pd.DataFrame, y=None):

        # 1. Определим параметры монтажа:

        grouped = X[X[LABEL_COL] != LABELS['Neutral']].groupby(self.sync_col)
        # - периодичность измерений – разность между соседними метками времени
        ts_delta = np.median((X[self.ts_col].shift(-1) - X[self.ts_col]).value_counts().index)

        # - среднее кол-во измерений на один (не нейтральный) жест
        self.ticks_per_gest = int(
            grouped[self.ts_col].count().median()
        )

        # - среднее кол-во измерений на один разделительный нейтральный жест
        self.ticks_per_nogo = int(
            ((grouped[self.ts_col].first() - grouped[self.ts_col].first().shift(1)) / ts_delta).median() - self.ticks_per_gest
        )

        # 2. Определим датчики с высоким уровнем сигнала

        omg_medians = pd.DataFrame(
            MinMaxScaler().fit_transform(pd.DataFrame(X[OMG_CH].median(axis=0))),
            index=OMG_CH, columns=['omg']
        )
        self.hi_val_sensors = omg_medians[omg_medians['omg'] > self.hi_val_threshold].index.to_list()

        # 3. Исключим датчики с внезапными падениями сигнала
        # (используем для этого заданный порог отношения первого процентиля к медиане)

        # По каждому из активных датчиков посчитаем отношение первого процентиля к медиане
        q_to_med = pd.Series(
            X[self.hi_val_sensors].quantile(0.01) / X[self.hi_val_sensors].median(),
            index=self.hi_val_sensors
        )
        # Отфильтруем датчики по заданному порогу self.sudden_drop_threshold
        sudden_drop_sensors = q_to_med[q_to_med <= self.sudden_drop_threshold].index
        self.mark_sensors = [sensor for sensor in self.hi_val_sensors if sensor not in sudden_drop_sensors]

        # 4. Исключим датчики с перегрузкой

        # Сколько идущих подряд максимальных значений считать перегрузкой
        in_a_row_threshold = 5
        # Доля перегруженного сигнала, чтобы исключить датчик из определения границ
        clip_threshold = 0.05

        clip_sensors = []

        # Для каждого из рассматриваемых датчиков найдем его максимум
        for sensor in self.mark_sensors:
            mask = X[sensor] == X[sensor].max()
            in_a_row = []
            cur = 0
            for x in mask:
                if not x:
                    if cur >= in_a_row_threshold:
                        in_a_row.append(cur)
                    cur = 0
                else:
                    cur += 1
            if cur >= in_a_row_threshold:
                in_a_row.append(cur)
            if sum(in_a_row) / X.shape[0] > clip_threshold:
                clip_sensors.append(sensor)

        if len(clip_sensors):
            self.mark_sensors = [sensor for sensor in self.mark_sensors if sensor not in clip_sensors]

        # Теперь у нас готов список датчиков self.mark_sensors, по которым мы и будем определять границы жестов

        # Установим ширину окна (для сглаживания медианой)
        self.window = self.ticks_per_gest // 3
        # и параметр spacing для вычисления градиентов
        self.spacing = self.ticks_per_gest // 3

        return self

    def transform(self, X: pd.DataFrame, y=None):
        if self.label_col in X.columns:
            X_marked, _ = self._mark(X)
            return X_marked
        else:
            return X.copy()


#Функция для чтения данных
def read_emg8(
        montage: str,
        dir: str = 'data',
        sep: str = ' ',
        drop_baseline_and_finish: bool = True,
        omg_only: bool = False
        ) -> pd.DataFrame:
    '''
    Осуществляет чтение файла с данными измерений монтажа .emg8.

    Добавляет в возвращаемый датафрейм признак `sample`, представляющий собой порядковый номер жеста в монтаже.

    ### Параметры

    **montage**: *str*<br>
    Имя файла для чтения

    **dir**: *str, default="data"*<br>
    Название папки, в которой находится файл

    **sep**: *str, default=' '*<br>
    Символ-разделитель в csv-файле

    **drop_baseline_and_finish**: *bool, default=True*<br>
    Удалять ли в начале и в конце монтажа измерения с метками `Baseline` и `Finish` соответственно

    **omg_only**: *bool, default=False*<br>
    Читать только столбцы датчиков OMG (для подгрузки тестовых данных)

    ### Возвращаемый результат

    **data**: *DataFrame*<br>
    Датафрейм с прочитанными данными
    '''
    path = os.path.join(dir, montage)
    cols = OMG_CH if omg_only else OMG_CH + [LABEL_COL, TS_COL]
    data = pd.read_csv(path, sep=sep, index_col=None)[cols]

    if not omg_only:
        if drop_baseline_and_finish:
            mask = (data[LABEL_COL] != LABELS['Finish']) & (data[LABEL_COL] != LABELS['Baseline'])
            data = data[mask].reset_index(drop=True)

        bounds = data[data[LABEL_COL] != data[LABEL_COL].shift(1)].index

        for i, lr in enumerate(zip(bounds, np.append(bounds[1:], [data.index[-1]]))):
            l, r = lr  # l, r - индексы начала текущей и следующей эпохи соответственно
            data.loc[l: r, SYNC_COL] = i

    return data

def split_data(data_marked, last_train_idx=5500):
    X = data_marked.drop(['act_label', 'id', 'ts', 'sample'], axis=1)
    y = data_marked['act_label']

    X_train = X[OMG_CH].values[:last_train_idx]
    y_train = y['act_label'].values[:last_train_idx]

    X_test = X[OMG_CH].values[last_train_idx:]
    y_test = y['act_label'].values[last_train_idx:]

    return X_train, X_test, y_train, y_test


# Функция для создания последовательностей
def create_sequences(data, labels, timesteps):
    X, y = [], []
    for i in range(len(data) - timesteps + 1):
        X.append(data[i:i + timesteps])
        y.append(labels[i + timesteps - 1])
    return np.array(X), np.array(y)

# Функция для создания последовательностей и кодирования меток
def prepare_sequences(X_train_array, X_test_array, y_train_array, y_test_array, timesteps):
    # Проверка и преобразование в np.ndarray
    if not isinstance(X_train_array, np.ndarray):
        X_train_array = np.array(X_train_array)
    if not isinstance(y_train_array, np.ndarray):
        y_train_array = np.array(y_train_array)
    if not isinstance(X_test_array, np.ndarray):
        X_test_array = np.array(X_test_array)
    if not isinstance(y_test_array, np.ndarray):
        y_test_array = np.array(y_test_array)

    X_train_seq, y_train_seq = create_sequences(X_train_array, y_train_array, timesteps)
    X_test_seq, y_test_seq = create_sequences(X_test_array, y_test_array, timesteps)

    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(y_train_seq.reshape(-1, 1))
    y_test_encoded = encoder.transform(y_test_seq.reshape(-1, 1))

    return X_train_seq, X_test_seq, y_train_encoded, y_test_encoded, encoder

# Функция для оценки модели с использованием кросс-валидации
def build_evaluate_model(timesteps, X_train_seq, y_train_encoded, X_test_seq, y_test_encoded):

    # Построение модели
    model = Sequential([
        LSTM(64, input_shape=(timesteps, X_train_seq.shape[2]), return_sequences=True),
        LayerNormalization(),
        Dropout(0.2),
        LSTM(128, return_sequences=True),
        LayerNormalization(),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        LayerNormalization(),
        Dense(32, activation='relu'),
        Dense(y_train_encoded.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Кросс-валидация
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    for train_index, val_index in kf.split(X_train_seq):
        X_train_fold, X_val_fold = X_train_seq[train_index], X_train_seq[val_index]
        y_train_fold, y_val_fold = y_train_encoded[train_index], y_train_encoded[val_index]

        model.fit(X_train_fold, y_train_fold, epochs=15, batch_size=32, verbose=0)
        val_loss, val_accuracy = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        accuracies.append(val_accuracy)

    print(f"Average validation accuracy: {np.mean(accuracies)}")

    # Обучение модели на всех тренировочных данных
    model.fit(X_train_seq, y_train_encoded, epochs=20, batch_size=32, verbose=0)

    # Оценка на тестовых данных
    test_loss, test_accuracy = model.evaluate(X_test_seq, y_test_encoded, verbose=0)
    print(f"Test accuracy: {test_accuracy}")

    return np.mean(accuracies), test_accuracy

def validate_timesteps(
        X_train, X_test, y_train, y_test,
        timesteps_values: list = [1, 2, 3],
        best_timesteps: int = None,
        best_val_accuracy: float = 0,
        best_test_accuracy: float = 0,

    ):

    for timesteps in timesteps_values:
        # Создание последовательностей и кодирование меток
        X_train_seq, X_test_seq, y_train_encoded, y_test_encoded, encoder = prepare_sequences(
                                                                                            X_train, X_test, y_train, y_test, timesteps
                                                                                        )
        val_accuracy, test_accuracy = build_evaluate_model(timesteps, X_train_seq, y_train_encoded, X_test_seq, y_test_encoded)
        print(f"Timesteps: {timesteps}, Validation Accuracy: {val_accuracy}, Test Accuracy: {test_accuracy}")
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_test_accuracy = test_accuracy
            best_timesteps = timesteps

    # print(f"Best Timesteps: {best_timesteps}, Best Validation Accuracy: {best_val_accuracy}, Best Test Accuracy: {best_test_accuracy}")

    return(best_timesteps)
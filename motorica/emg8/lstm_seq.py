import numpy as np
import optuna
import tensorflow_addons as tfa
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.utils import to_categorical
from motorica.emg8.constants import *

def create_sequences(X, y, timesteps=2):
    # Создание последовательностей
    X_sequences, y_sequences = [], []
    for i in range(len(X) - timesteps):
        X_sequences.append(X[i:i+timesteps])
        y_sequences.append(y[i+timesteps])

    return np.array(X_sequences), np.array(y_sequences)

def prepare_seq(X_train, y_train, X_test, y_test, timesteps):
    # Преобразование в numpy массивы
    X_sequences_train, y_sequences_train = create_sequences(X_train, y_train, timesteps)
    X_sequences_test, y_sequences_test = create_sequences(X_test, y_test, timesteps)

    # Приведение к формату (samples, timesteps, features)
    X_sequences_train = X_sequences_train.reshape((X_sequences_train.shape[0], timesteps, X_sequences_train.shape[2]))
    X_sequences_test = X_sequences_test.reshape((X_sequences_test.shape[0], timesteps, X_sequences_test.shape[2]))

    # Преобразование меток в one-hot encoding
    y_sequences_train_encoded = to_categorical(y_sequences_train)
    y_sequences_test_encoded = to_categorical(y_sequences_test)
    num_classes = to_categorical(y_sequences_test).shape[1]

    return X_sequences_train, y_sequences_train_encoded, X_sequences_test, y_sequences_test_encoded, num_classes


# Функция для создания модели LSTM
def create_lstm_model(input_shape, n_units=32, dropout_rate=0.2, learning_rate=0.001, use_bidirectional=False, use_gru=False):
    model = Sequential()
    if use_gru:
        model.add(GRU(32, input_shape=input_shape, return_sequences=True, kernel_regularizer=l2(0.01)))
    elif use_bidirectional:
        model.add(Bidirectional(LSTM(32, input_shape=input_shape, return_sequences=True, kernel_regularizer=l2(0.01))))
    else:
        model.add(LSTM(32, input_shape=input_shape, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(dropout_rate))
    if use_gru:
        model.add(GRU(n_units, return_sequences=False, kernel_regularizer=l2(0.01)))
    elif use_bidirectional:
        model.add(Bidirectional(LSTM(n_units, return_sequences=False, kernel_regularizer=l2(0.01))))
    else:
        model.add(LSTM(n_units, return_sequences=False, kernel_regularizer=l2(0.01)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=[tfa.metrics.F1Score(num_classes=NUM_CLASSES, average='weighted')])
    return model



# Функция для оптимизации гиперпараметров
def optimize_lstm(X_train_combined, y_train_combined, X_val, y_val):
    def objective(trial):
        timesteps = trial.suggest_int('timesteps', 2, 4)
        n_units = trial.suggest_int('n_units', 32, 128)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.2)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)
        use_bidirectional = trial.suggest_categorical('use_bidirectional', [True, False])
        use_gru = trial.suggest_categorical('use_gru', [True, False])

        X_train_seq, y_train_seq, X_val_seq, y_val_seq, num_classes = prepare_seq(X_train_combined, y_train_combined, X_val, y_val, timesteps)

        model = create_lstm_model((timesteps, X_train_seq.shape[2]), n_units, dropout_rate, learning_rate, use_bidirectional, use_gru)
        early_stopping = EarlyStopping(monitor='val_f1_score', patience=10, restore_best_weights=True, mode='max')
        model.fit(X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq), epochs=80, batch_size=64, callbacks=[early_stopping], verbose=0)
        val_loss, val_f1_score = model.evaluate(X_val_seq, y_val_seq, verbose=0)
        return val_f1_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    return study.best_params

def preprocessing(buffer, timesteps):
    # Преобразование буфера в последовательность для модели
    buffer_array = np.array(buffer)
    if buffer_array.shape[0] == timesteps:
        # Создание последовательностей
        sequences = []
        for i in range(len(buffer_array) - timesteps + 1):
            seq = buffer_array[i:i+timesteps]
            sequences.append(seq)
        sequences = np.array(sequences)

        # Приведение к формату (samples, timesteps, features)
        sequences = sequences.reshape((sequences.shape[0], timesteps, len(OMG_CH)))

        sample_preprocessed = sequences
        return sample_preprocessed
    else:
        return None

# Функция для выполнения инференса (предсказания)
def inference(model, X_seq):
    y_pred_prob = model.predict(X_seq)
    return y_pred_prob

def postprocessing(y_pred_prob):
    y_pred = np.argmax(y_pred_prob, axis=1)[0]  # Получаем скалярное значение
    return y_pred
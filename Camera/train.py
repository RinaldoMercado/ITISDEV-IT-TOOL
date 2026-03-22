#train.py
import os
import numpy as np

# Training script for the ManuMano LSTM gesture classifier.
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = 'MP_Data'
SEQUENCE_LENGTH = 30

ACTIONS = [d for d in os.listdir(DATA_PATH)
           if os.path.isdir(os.path.join(DATA_PATH, d))]

# ---------------------------- Load Data ----------------------------
X, y = [], []
label_map = {label:num for num,label in enumerate(ACTIONS)}

for action in ACTIONS:
    action_path = os.path.join(DATA_PATH, action)
    files = sorted(os.listdir(action_path))

    # Group frames per sequence
    for i in range(0, len(files), SEQUENCE_LENGTH):
        sequence = []
        for j in range(SEQUENCE_LENGTH):
            frame = np.load(os.path.join(action_path, files[i+j]))
            sequence.append(frame)

        X.append(sequence)
        y.append(label_map[action])

X = np.array(X)
y = to_categorical(y).astype(int)

# ---------------------------- Train/Test Split ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

# ---------------------------- Optimized Model ----------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, X.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dense(64, activation='relu'),
    Dense(len(ACTIONS), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

model.summary()

# ---------------------------- Train ----------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(
    X_train, y_train,
    epochs=80,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

model.save('action.h5')
print("Model saved as action.h5")


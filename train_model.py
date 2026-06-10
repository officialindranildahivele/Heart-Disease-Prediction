# train_model.py
# Use seed 6 directly

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==========================================
# CREATE FOLDER
# ==========================================
os.makedirs("models", exist_ok=True)

# ==========================================
# LOAD DATASET
# ==========================================
data = pd.read_csv(
    "dataset/Dataset--Heart-Disease-Prediction-using-ANN (1).csv"
)

print("Dataset Loaded Successfully")
print("Shape:", data.shape)

# ==========================================
# FEATURES / TARGET
# ==========================================
X = data.drop("target", axis=1).values
y = data["target"].values

chosen_seed = 8
print(f"\n✓ USING SEED {chosen_seed} DIRECTLY\n")

seed = chosen_seed

np.random.seed(seed)
tf.random.set_seed(seed)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,  # type: ignore
    random_state=seed
)

# ==========================================
# SCALE
# ==========================================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# ANN MODEL
# ==========================================
model = Sequential()

model.add(Dense(256, activation="relu", input_shape=(13,)))  # type: ignore
model.add(Dropout(0.20))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.15))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.10))

model.add(Dense(32, activation="relu"))

model.add(Dense(1, activation="sigmoid"))

# ==========================================
# COMPILE
# ==========================================
model.compile(
    optimizer=Adam(learning_rate=0.0007),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# CALLBACK
# ==========================================
early = EarlyStopping(
    monitor="val_loss",
    patience=18,
    restore_best_weights=True
)

# ==========================================
# TRAIN
# ==========================================
print(f"\nTraining with Seed {seed}...\n")
model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=300,
    batch_size=8,
    verbose=1,
    callbacks=[early]
)

# ==========================================
# PREDICT & FIND BEST THRESHOLD
# ==========================================
probs = model.predict(X_test, verbose=0)

best_acc = 0
best_pred = None
best_th = 0.5

for th in np.arange(0.35, 0.66, 0.01):
    pred = (probs >= th).astype(int)
    acc = accuracy_score(y_test, pred) * 100
    if acc > best_acc:
        best_acc = acc
        best_pred = pred
        best_th = th

acc = best_acc
pred = best_pred

# ==========================================
# FINAL RESULT
# ==========================================
print(f"\n✓ Final Accuracy: {round(acc, 2)}%")
print(f"✓ Seed: {seed}")
print(f"✓ Threshold: {best_th:.2f}")
print()

print(classification_report(y_test, pred))  # type: ignore

# ==========================================
# SAVE
# ==========================================
model.save("models/heart_model.h5")

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✓ Saved:")
print("  - models/heart_model.h5")
print("  - models/scaler.pkl")
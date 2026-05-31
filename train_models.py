import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor, StackingRegressor
from sklearn.metrics import r2_score, f1_score

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Классификация
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ==== РЕГРЕССИЯ (CARS) ====
print("=" * 50)
print("Обучение моделей РЕГРЕССИИ (cars)")
print("=" * 50)

# Загрузка и предобработка данных
cars_df = pd.read_csv('data/cars.csv')
# Удалим строки с пропусками, если есть
cars_df = cars_df.dropna()

X_cars = cars_df.drop('price_usd', axis=1)
y_cars = cars_df['price_usd']

# Разделение
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cars, y_cars, test_size=0.2, random_state=42
)

# Определение категориальных и числовых признаков
cat_features = ['brand', 'model', 'transmission', 'body_type', 'fuel_type', 'color']
num_features = ['year', 'mileage', 'engine_capacity', 'doors']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ]
)

os.makedirs('models_regression', exist_ok=True)

# ML1: Классическая модель — Ridge Regression
ridge = Pipeline([
    ('prep', preprocessor),
    ('model', Ridge(alpha=1.0))
])
ridge.fit(X_train_c, y_train_c)
r2_ridge = r2_score(y_test_c, ridge.predict(X_test_c))
print(f"ML1 Ridge R2: {r2_ridge:.4f}")
with open('models_regression/ridge.pkl', 'wb') as f:
    pickle.dump(ridge, f)

# ML2: Ансамблевая — XGBoost (бустинг)
xgb_reg = Pipeline([
    ('prep', preprocessor),
    ('model', XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=4))
])
xgb_reg.fit(X_train_c, y_train_c)
r2_xgb = r2_score(y_test_c, xgb_reg.predict(X_test_c))
print(f"ML2 XGBoost R2: {r2_xgb:.4f}")
with open('models_regression/xgboost.pkl', 'wb') as f:
    pickle.dump(xgb_reg, f)

# ML3: CatBoost
cb_reg = CatBoostRegressor(
    cat_features=cat_features,
    iterations=300,
    depth=6,
    learning_rate=0.1,
    verbose=0,
    random_state=42
)
cb_reg.fit(X_train_c, y_train_c)
r2_cb = r2_score(y_test_c, cb_reg.predict(X_test_c))
print(f"ML3 CatBoost R2: {r2_cb:.4f}")
cb_reg.save_model('models_regression/catboost.cbm')

# ML4: Bagging (бэггинг)
bag_reg = Pipeline([
    ('prep', preprocessor),
    ('model', BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=10),
        n_estimators=50,
        random_state=42,
        n_jobs=4
    ))
])
bag_reg.fit(X_train_c, y_train_c)
r2_bag = r2_score(y_test_c, bag_reg.predict(X_test_c))
print(f"ML4 Bagging R2: {r2_bag:.4f}")
with open('models_regression/bagging.pkl', 'wb') as f:
    pickle.dump(bag_reg, f)

# ML5: Stacking (стэкинг)
estimators_stack = [
    ('ridge', Ridge(alpha=1.0)),
    ('svr', SVR(kernel='rbf', C=1.0, gamma='scale')),
    ('tree', DecisionTreeRegressor(max_depth=10))
]
stack_reg_inner = StackingRegressor(
    estimators=estimators_stack,
    final_estimator=Ridge(alpha=1.0),
    passthrough=False,
    n_jobs=4
)
stack_reg = Pipeline([
    ('prep', preprocessor),
    ('model', stack_reg_inner)
])
stack_reg.fit(X_train_c, y_train_c)
r2_stack = r2_score(y_test_c, stack_reg.predict(X_test_c))
print(f"ML5 Stacking R2: {r2_stack:.4f}")
with open('models_regression/stacking.pkl', 'wb') as f:
    pickle.dump(stack_reg, f)

# ML6: Нейронная сеть (TensorFlow/Keras)
# Подготовка данных для нейросети
X_train_c_processed = preprocessor.fit_transform(X_train_c)
X_test_c_processed = preprocessor.transform(X_test_c)

nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_c_processed.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
nn_model.fit(
    X_train_c_processed, y_train_c,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)
r2_nn = r2_score(y_test_c, nn_model.predict(X_test_c_processed, verbose=0).flatten())
print(f"ML6 Neural Network R2: {r2_nn:.4f}")
nn_model.save('models_regression/neural_network.keras')

# Сохраняем preprocessor для нейросети
with open('models_regression/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

print("\nМодели регрессии сохранены в папку models_regression/")


# ==== КЛАССИФИКАЦИЯ (DIABETES) ====
print("\n" + "=" * 50)
print("Обучение моделей КЛАССИФИКАЦИИ (diabetes)")
print("=" * 50)

diabetes_df = pd.read_csv('data/diabetes.csv')
diabetes_df = diabetes_df.dropna()

X_d = diabetes_df.drop('Diabetes_012', axis=1)
y_d = diabetes_df['Diabetes_012']

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42, stratify=y_d
)

scaler_d = StandardScaler()
X_train_d_scaled = scaler_d.fit_transform(X_train_d)
X_test_d_scaled = scaler_d.transform(X_test_d)

os.makedirs('models_classification', exist_ok=True)

# ML1: Logistic Regression
logreg = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42)
logreg.fit(X_train_d_scaled, y_train_d)
f1_lr = f1_score(y_test_d, logreg.predict(X_test_d_scaled), average='weighted')
print(f"ML1 LogisticRegression F1: {f1_lr:.4f}")
with open('models_classification/logreg.pkl', 'wb') as f:
    pickle.dump(logreg, f)

# ML2: XGBoost (бустинг)
xgb_clf = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    objective='multi:softmax', num_class=3,
    random_state=42, n_jobs=4
)
xgb_clf.fit(X_train_d, y_train_d)
f1_xgb = f1_score(y_test_d, xgb_clf.predict(X_test_d), average='weighted')
print(f"ML2 XGBoost F1: {f1_xgb:.4f}")
with open('models_classification/xgboost.pkl', 'wb') as f:
    pickle.dump(xgb_clf, f)

# ML3: CatBoost
cb_clf = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.1,
    loss_function='MultiClass',
    classes_count=3,
    verbose=0,
    random_state=42
)
cb_clf.fit(X_train_d, y_train_d)
f1_cb = f1_score(y_test_d, cb_clf.predict(X_test_d), average='weighted')
print(f"ML3 CatBoost F1: {f1_cb:.4f}")
cb_clf.save_model('models_classification/catboost.cbm')

# ML4: Bagging (бэггинг)
bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=10),
    n_estimators=50,
    random_state=42,
    n_jobs=4
)
bag_clf.fit(X_train_d_scaled, y_train_d)
f1_bag = f1_score(y_test_d, bag_clf.predict(X_test_d_scaled), average='weighted')
print(f"ML4 Bagging F1: {f1_bag:.4f}")
with open('models_classification/bagging.pkl', 'wb') as f:
    pickle.dump(bag_clf, f)

# ML5: Stacking (стэкинг)
estimators_stack_clf = [
    ('lr', LogisticRegression(max_iter=500, solver='lbfgs')),
    ('svc', SVC(kernel='rbf', probability=True)),
    ('tree', DecisionTreeClassifier(max_depth=10))
]
stack_clf = StackingClassifier(
    estimators=estimators_stack_clf,
    final_estimator=LogisticRegression(max_iter=500, solver='lbfgs'),
    passthrough=False,
    n_jobs=4
)
stack_clf.fit(X_train_d_scaled, y_train_d)
f1_stack = f1_score(y_test_d, stack_clf.predict(X_test_d_scaled), average='weighted')
print(f"ML5 Stacking F1: {f1_stack:.4f}")
with open('models_classification/stacking.pkl', 'wb') as f:
    pickle.dump(stack_clf, f)

# ML6: Нейронная сеть (TensorFlow/Keras)
num_classes = 3
y_train_d_cat = keras.utils.to_categorical(y_train_d, num_classes)
y_test_d_cat = keras.utils.to_categorical(y_test_d, num_classes)

nn_clf = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_d_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])
nn_clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop2 = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
nn_clf.fit(
    X_train_d_scaled, y_train_d_cat,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop2],
    verbose=0
)
f1_nn = f1_score(y_test_d, np.argmax(nn_clf.predict(X_test_d_scaled, verbose=0), axis=1), average='weighted')
print(f"ML6 Neural Network F1: {f1_nn:.4f}")
nn_clf.save('models_classification/neural_network.keras')

with open('models_classification/scaler.pkl', 'wb') as f:
    pickle.dump(scaler_d, f)

print("\nМодели классификации сохранены в папку models_classification/")

# ==== Общий скорборд ====
print("\n" + "=" * 50)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("=" * 50)
print("\nРЕГРЕССИЯ (R2):")
print(f"  1. Ridge:       {r2_ridge:.4f}")
print(f"  2. XGBoost:     {r2_xgb:.4f}")
print(f"  3. CatBoost:    {r2_cb:.4f}")
print(f"  4. Bagging:     {r2_bag:.4f}")
print(f"  5. Stacking:    {r2_stack:.4f}")
print(f"  6. Neural Net:  {r2_nn:.4f}")
print("\nКЛАССИФИКАЦИЯ (F1 weighted):")
print(f"  1. LogReg:      {f1_lr:.4f}")
print(f"  2. XGBoost:     {f1_xgb:.4f}")
print(f"  3. CatBoost:    {f1_cb:.4f}")
print(f"  4. Bagging:     {f1_bag:.4f}")
print(f"  5. Stacking:    {f1_stack:.4f}")
print(f"  6. Neural Net:  {f1_nn:.4f}")

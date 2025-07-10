import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

# Klasik makine öğrenmesi algoritmaları (KNN, SVM, RF, GBC, MLP)
# Scikit-learn dökümantasyonu:
# https://scikit-learn.org/stable/supervised_learning.html



X_train = np.load('results/X_train.npy')
X_test  = np.load('results/X_test.npy')
y_train = np.load('results/y_train.npy')
y_test  = np.load('results/y_test.npy')




configs = {
    'knn': (KNeighborsClassifier(), {'n_neighbors':[5,7]}),
    'svm': (SVC(),                  {'C':[0.1,1], 'gamma':[0.01]}),
    'rf':  (RandomForestClassifier(random_state=42),{'n_estimators':[100], 'max_depth':[None,20]}),
    'gbc': (GradientBoostingClassifier(random_state=42),{'n_estimators':[100], 'learning_rate':[0.1]}),
    'mlp': (MLPClassifier(max_iter=300, random_state=42),{'hidden_layer_sizes':[(100,)], 'alpha':[1e-4]})
}



results = []
os.makedirs('models', exist_ok=True)
for name, (model, params) in configs.items():
    path = f'models/{name}_method1.pkl'
    if os.path.exists(path):
        print(f"{name.upper()} modeli hazır, yüklendi.")
        clf = joblib.load(path)
        best_params = clf.get_params()
    else:
        print(f"{name.upper()} eğitiliyor...")
        gs = GridSearchCV(
            model, params,
            cv=2, n_jobs=-1, verbose=1
        )
        gs.fit(X_train, y_train)
        clf = gs.best_estimator_
        best_params = gs.best_params_
        joblib.dump(clf, path)


    y_pred = clf.predict(X_test)
    acc    = (y_pred == y_test).mean()
    prec   = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec    = recall_score(   y_test, y_pred, average='macro', zero_division=0)
    f1m    = f1_score(       y_test, y_pred, average='macro', zero_division=0)

    results.append({
        'model'    : name,
        'params'   : best_params,
        'accuracy' : round(acc,4),
        'precision': round(prec,4),
        'recall'   : round(rec,4),
        'f1'       : round(f1m,4)
    })



df = pd.DataFrame(results)
os.makedirs('results', exist_ok=True)
df.to_csv('results/method1_results.csv', index=False)
print(df)


import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

X_test = np.load('results/X_test.npy')
y_test = np.load('results/y_test.npy')

classifiers = ['knn','svm','rf','gbc','mlp']
os.makedirs('results', exist_ok=True)



for key in classifiers:
    model_file = f'models/{key}_method1.pkl'
    if not os.path.exists(model_file):
        continue

    clf = joblib.load(model_file)
    y_pred = clf.predict(X_test)

    print(f"\n*** {key.upper()} Classification Report ***")
    print(classification_report(y_test, y_pred, zero_division=0))

    fig, ax = plt.subplots(figsize=(10,8))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        normalize='true',
        cmap='Blues',
        values_format='.2f',
        ax=ax
    )
    
    
    ax.set_xlabel('Tahmin', fontsize=12)
    ax.set_ylabel('Gerçek', fontsize=12)
    ax.set_title(f'{key.upper()} CM (normalize)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'results/{key}_cm.png', dpi=150)
    plt.show()
    plt.close(fig)


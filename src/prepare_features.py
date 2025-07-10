import os, sys
import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from features import compute_features
from data_loader import get_annotations, preprocess_image






CSV_FILE    = 'annotations_subset.csv'
IMG_DIR     = 'data/tsinghua_subset'
TOP_K       = 15
MIN_PER_CL  = 5
MAX_SAMPLES = 10000
TEST_SIZE   = 0.2
SEED        = 42




OUT_DIR      = 'results'
XTR_PATH     = os.path.join(OUT_DIR, 'X_train.npy')
XTE_PATH     = os.path.join(OUT_DIR, 'X_test.npy')
YTR_PATH     = os.path.join(OUT_DIR, 'y_train.npy')
YTE_PATH     = os.path.join(OUT_DIR, 'y_test.npy')
SCALER_PATH  = 'models/scaler.pkl'
PCA_PATH     = 'models/pca.pkl'





required = [XTR_PATH, XTE_PATH, YTR_PATH, YTE_PATH, SCALER_PATH, PCA_PATH]
if all(os.path.exists(p) for p in required):
    print("Özellikler zaten oluşturulmuş, prepare_features çağrısı atlanıyor.")
    sys.exit(0)



df = get_annotations(CSV_FILE)
top_labels = df['label'].value_counts().nlargest(TOP_K).index
df = df[df['label'].isin(top_labels)]
counts = df['label'].value_counts()
valid = counts[counts >= MIN_PER_CL].index
df = df[df['label'].isin(valid)].reset_index(drop=True)
print(f"Sınıf sayısı: {len(valid)}")



n = min(MAX_SAMPLES, len(df))
df = df.sample(n=n, random_state=SEED).reset_index(drop=True)
print(f"Örnek sayısı: {n}")




features = []
labels   = []
for _, row in df.iterrows():
    img = preprocess_image(row)
    fv  = compute_features(img)
    features.append(fv)
    labels.append(row['label'])

X = np.array(features, dtype=np.float32)
y = np.array(labels)
print("Öznitelik matrisi boyutu:", X.shape)




scaler = StandardScaler()
Xs     = scaler.fit_transform(X)
pca    = PCA(n_components=0.95, random_state=SEED)
Xp     = pca.fit_transform(Xs)
print("PCA sonrası boyut:", Xp.shape)

# PCA (Principal Component Analysis) uygulaması
# Kaynak: Jolliffe, I. T. (2002). "Principal Component Analysis". Springer Series in Statistics.
# DOI: https://doi.org/10.1007/b98835



X_tr, X_te, y_tr, y_te = train_test_split(
    Xp, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=SEED
)
print("Train/Test şekli:", X_tr.shape, X_te.shape)




os.makedirs(OUT_DIR, exist_ok=True)
np.save(XTR_PATH, X_tr)
np.save(XTE_PATH, X_te)
np.save(YTR_PATH, y_tr)
np.save(YTE_PATH, y_te)
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(pca, PCA_PATH)

print("Özellik hazırlığı tamamlandı ve kaydedildi.")


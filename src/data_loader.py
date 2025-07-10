import os
import cv2
import pandas as pd




ANNOT_FILE = 'annotations_subset.csv'
IMG_FOLDER = 'data/tsinghua_subset'


def get_annotations(csv_path=ANNOT_FILE):
    df = pd.read_csv(csv_path)
    return df[['image_path','xmin','ymin','xmax','ymax','label']]





def preprocess_image(row):
    fname = os.path.basename(row['image_path'])
    full = os.path.join(IMG_FOLDER, fname)
    img = cv2.imread(full)
    
    if img is None:
        raise IOError(f"Image bulunamadı: {full}")

    x1, y1, x2, y2 = map(int, (
        row['xmin'], row['ymin'], row['xmax'], row['ymax']
    ))
    h, w = img.shape[:2]
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(w,x2), min(h,y2)

    if x2 <= x1 or y2 <= y1:
        crop = img
    else:
        crop = img[y1:y2, x1:x2]

    return cv2.resize(crop, (224,224))


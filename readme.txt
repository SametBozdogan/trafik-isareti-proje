# Trafik İşareti Sınıflandırma Projesi

Öğrenci : Samet Bozdoğan – 211101009
Ders    : YAP 470
Dönem   : 2024-25 Yaz




## Proje Kapsamı

Bu proje kapsamında, görüntü işleme ve makine öğrenmesi yöntemleriyle trafik işaretlerini sınıflandıran bir sistem geliştirildi. Kullanılan ana veriseti: **Tsinghua-Tencent 100K**'dan alınmış 10.000 örnek içeren bir alt kümedir. Kodlar ve eğitim süreçleri Python ile yazılmıştır; değerlendirme ve görselleştirme adımları ise Jupyter Notebook üzerinde yapılmıştır.


## Önemli Hususlar

data/ ve models/ dosyası büyük dosya olduğundan ötürü Githuba yüklenmedi. Google drive linki aşağıdadır. data/ ve models/ klasörleri indirildikten sonra ana dizine yerleştirilmelidir. Aşağıdaki repo yapısını inceleyebilirsiniz.
https://drive.google.com/drive/folders/1VhDlE-BbuDzy9bvbqEo550UJQf9BrOPb?usp=sharing

## Proje Amacı

Gerçek yol görüntülerindeki trafik işaretlerini otomatik tanımayı amaçlıyorum.
İlk yöntem (“Method 1”) klasik makine öğrenmesi tekniklerini (KNN, SVM-RBF, RF, GBC, MLP) HOG+HSV+PCA öznitelikleriyle karşılaştırdı.





## Repo Yapısı


project_root/
├── data/
│   └── tsinghua_subset/
├── models/...
├── notebooks/
│   ├── method1_training.ipynb
│   └── method1_evaluation.ipynb
├── results/...
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── prepare_features.py
│   ├── train_method1.py
│   └── eval_method1.py
├── annotations_subset.csv
├── requirements.txt
└── readme.txt






## Kurulum & Çalıştırma ## 

1. Repoyu klonlayın:

   git clone https://github.com/SametBozdogan/trafik-isareti-proje.git
   cd trafik-isareti-proje

2. Sanal ortam oluşturup paketleri yükleyin:

   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

3. Öznitelik hazırlama & modele girdi üretme:

   python src/prepare_features.py

4. Model eğitimi & karşılaştırma (Method 1):

   python src/train_method1.py

5. Test setinde değerlendirme & confusion matrix’ler:

   python src/eval_method1.py

   

   

   
   ## Jupyter Notebook’lar ##

   ana dizinde:
   jupyter notebook

   

   notebooks/method1_training.ipynb – Veri hazırlama, öznitelik çıkarma ve tüm klasik makine öğrenmesi modellerinin eğitimini adım adım yapan ana notebook dosyasıdır. Hazırlanan modeller models/ klasörüne kaydedilir. Ara çıktılar ve özet tablo sonuçları notebook üzerinde gösterilir.

   

   notebooks/method1_evaluation.ipynb – Eğitilmiş modellerin test seti üzerindeki performansını detaylı olarak raporlayan ve confusion matrix görsellerini üreten değerlendirme notebook dosyasıdır. Precision, recall, f1-score gibi metrikler tablo halinde, karışıklık matrisi ise görsel olarak gösterilir.

   

   
   

   ## Kullanılan Kaynaklar

   Tsinghua-Tencent 100K Dataset: https://cg.cs.tsinghua.edu.cn/traffic-sign/
   HOG: Dalal \& Triggs (2005)
   PCA: Jolliffe (2002)


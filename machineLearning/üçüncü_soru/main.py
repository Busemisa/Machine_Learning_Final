import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Rastgelelik için sabit seed
np.random.seed(42)

# Veriyi yükleme
try:
    data = pd.read_csv('../diabetes.csv')  # Dosya yolunu belirtiniz
except FileNotFoundError:
    print("Dosya bulunamadı. Lütfen dosya yolunu kontrol edin.")
    exit()

# Özellikler ve etiketlerin ayrılması
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Veriyi %70 eğitim, %30 test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# En iyi k değerini belirlemek için k değerlerini test etme
k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

best_k = k_range[np.argmax(k_scores)]
print(f"En iyi k değeri: {best_k}")

# En iyi k değeri ile KNN modelini eğitme
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Tahminlerin yapılması
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Performans metriklerinin hesaplanması
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

train_precision = precision_score(y_train, y_train_pred, average='weighted')
test_precision = precision_score(y_test, y_test_pred, average='weighted')

train_recall = recall_score(y_train, y_train_pred, average='weighted')
test_recall = recall_score(y_test, y_test_pred, average='weighted')

train_f1 = f1_score(y_train, y_train_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

# Karışıklık matrislerinin hesaplanması
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

# Sonuçların CSV olarak kaydedilmesi
results = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Train': [train_accuracy, train_precision, train_recall, train_f1],
    'Test': [test_accuracy, test_precision, test_recall, test_f1]
}
results_df = pd.DataFrame(results)
results_df.to_csv('classification_report_knn.csv', index=False)

# Karışıklık matrislerini CSV olarak kaydetme
train_cm_df = pd.DataFrame(train_cm)
train_cm_df.to_csv('train_confusion_matrix_knn.csv', index=False)

test_cm_df = pd.DataFrame(test_cm)
test_cm_df.to_csv('test_confusion_matrix_knn.csv', index=False)

# Karışıklık matrislerinin grafikle gösterilmesi
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Eğitim Verisi Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')

plt.subplot(1, 2, 2)
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Test Verisi Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')

plt.tight_layout()
plt.savefig('confusion_matrices_knn.png')
plt.show()

# ROC eğrisinin çizilmesi ve AUC skorunun hesaplanması
y_test_proba = knn.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve_knn.png')
plt.show()

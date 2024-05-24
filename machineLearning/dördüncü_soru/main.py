import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
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

# MLP Modelinin oluşturulması ve eğitilmesi
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)

# SVM Modelinin oluşturulması ve eğitilmesi
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)

# MLP Modeli ile tahminlerin yapılması
y_train_pred_mlp = mlp.predict(X_train)
y_test_pred_mlp = mlp.predict(X_test)

# SVM Modeli ile tahminlerin yapılması
y_train_pred_svm = svm.predict(X_train)
y_test_pred_svm = svm.predict(X_test)

# Performans metriklerinin hesaplanması
def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

train_metrics_mlp = compute_metrics(y_train, y_train_pred_mlp)
test_metrics_mlp = compute_metrics(y_test, y_test_pred_mlp)

train_metrics_svm = compute_metrics(y_train, y_train_pred_svm)
test_metrics_svm = compute_metrics(y_test, y_test_pred_svm)

# Karışıklık matrislerinin hesaplanması
train_cm_mlp = confusion_matrix(y_train, y_train_pred_mlp)
test_cm_mlp = confusion_matrix(y_test, y_test_pred_mlp)

train_cm_svm = confusion_matrix(y_train, y_train_pred_svm)
test_cm_svm = confusion_matrix(y_test, y_test_pred_svm)

# Sonuçların CSV olarak kaydedilmesi
results = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Train_MLP': train_metrics_mlp,
    'Test_MLP': test_metrics_mlp,
    'Train_SVM': train_metrics_svm,
    'Test_SVM': test_metrics_svm
}
results_df = pd.DataFrame(results)
results_df.to_csv('classification_report_mlp_svm.csv', index=False)

# Karışıklık matrislerini CSV olarak kaydetme
train_cm_mlp_df = pd.DataFrame(train_cm_mlp)
train_cm_mlp_df.to_csv('train_confusion_matrix_mlp.csv', index=False)

test_cm_mlp_df = pd.DataFrame(test_cm_mlp)
test_cm_mlp_df.to_csv('test_confusion_matrix_mlp.csv', index=False)

train_cm_svm_df = pd.DataFrame(train_cm_svm)
train_cm_svm_df.to_csv('train_confusion_matrix_svm.csv', index=False)

test_cm_svm_df = pd.DataFrame(test_cm_svm)
test_cm_svm_df.to_csv('test_confusion_matrix_svm.csv', index=False)

# Karışıklık matrislerinin grafikle gösterilmesi
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
sns.heatmap(train_cm_mlp, annot=True, fmt='d', cmap='Blues')
plt.title('MLP Eğitim Verisi Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')

plt.subplot(2, 2, 2)
sns.heatmap(test_cm_mlp, annot=True, fmt='d', cmap='Blues')
plt.title('MLP Test Verisi Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')

plt.subplot(2, 2, 3)
sns.heatmap(train_cm_svm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Eğitim Verisi Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')

plt.subplot(2, 2, 4)
sns.heatmap(test_cm_svm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Test Verisi Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')

plt.tight_layout()
plt.savefig('confusion_matrices_mlp_svm.png')
plt.show()

# ROC eğrisinin çizilmesi ve AUC skorunun hesaplanması
y_test_proba_mlp = mlp.predict_proba(X_test)[:, 1]
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_test, y_test_proba_mlp)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

y_test_proba_svm = svm.predict_proba(X_test)[:, 1]
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, y_test_proba_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

plt.figure()
plt.plot(fpr_mlp, tpr_mlp, color='darkorange', lw=2, label=f'MLP ROC curve (area = {roc_auc_mlp:.2f})')
plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label=f'SVM ROC curve (area = {roc_auc_svm:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve_mlp_svm.png')
plt.show()

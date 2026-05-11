import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC

# --- ميثودات المعالجة اليدوية ---

# بتحول الصورة من ألوان لدرجات الرمادي (أبيض وأسود) عن طريق حساب متوسط الألوان لكل بكسل
def manual_grayscale(img):
    row, col, lay = img.shape
    gray_img = np.zeros((row, col), dtype=np.uint8)
    for i in range(row):
        for j in range(col):
            r = int(img[i, j, 2])
            g = int(img[i, j, 1])
            b = int(img[i, j, 0])
            avg = (r + g + b) // 3
            gray_img[i, j] = avg
    return gray_img

# بتتحكم في سطوع الصورة، بتجمع قيمة معينة لكل بكسل عشان تفتحها أو تغمقها مع التأكد إنها ما تطلع عن حدود (0-255)
def manual_brightness(img, value):
    row, col = img.shape
    new_img = np.zeros((row, col), dtype=np.uint8)
    for i in range(row):
        for j in range(col):
                val = int(img[i, j]) + value
                if val > 255: val = 255
                if val < 0: val = 0
                new_img[i, j] = val
    return new_img

# بتعمل "تغبيش" أو تنعيم للصورة عشان تخفف من حدة التفاصيل أو النويز باستخدام (Kernel) بتوزع القيم بالتساوي
def manual_blur(img_gray):
    row, col = img_gray.shape
    new_img = np.zeros((row, col))
    kernel = np.array([[1/9, 1/9, 1/9],
                       [1/9, 1/9, 1/9],
                       [1/9, 1/9, 1/9]])
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            region = img_gray[i - 1:i + 2, j - 1:j + 2]
            new_img[i, j] = np.sum(region * kernel)
    return new_img.astype(np.uint8)

# بتزيد من حدة تفاصيل الصورة (Sharpening) عشان تبرز الحواف والملامح اللي ممكن تكون ضاعت في الخطوات السابقة
def manual_sharpen(img_gray):
    row, col = img_gray.shape
    new_img = np.zeros((row, col))
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            region = img_gray[i - 1:i + 2, j - 1:j + 2]
            new_img[i, j] = np.clip(np.sum(region * kernel), 0, 255)
    return new_img.astype(np.uint8)

# بتعد تكرار كل درجة لون (من 0 لـ 255) في الصورة عشان نستخدمها كـ "خصائص" (Features) بتمثل الصورة رياضياً
def get_hist_features(img_gray):
    row, col = img_gray.shape
    hist = np.zeros(256)
    for i in range(row):
        for j in range(col):
            val = int(img_gray[i, j])
            hist[val] += 1
    return hist

# --- تحميل ومعالجة البيانات ---
lbl = []
datasetNormal = []
datasetEnhanced = []

classes = os.listdir("train")

# بنبلش نقرا الصور من الملفات
for cls in classes:
    classPath = os.path.join("train", cls)
    for imgName in os.listdir(classPath):
        imgPath = os.path.join(classPath, imgName)
        img = cv2.imread(imgPath)
        if img is None: continue

        # توحيد مقاس كل الصور عشان نقدر نقارن بينهم
        img = cv2.resize(img, (224, 224))
        imgGray = manual_grayscale(img)

        # 1. الداتا العادية: بنحول الصورة لفيكتور من البكسلات (بدون أي تعديل إضافي)
        datasetNormal.append(imgGray.flatten())

        # 2. الداتا المحسنة: بنمرر الصورة بسلسلة فلاتر (تفتيح، تنعيم، حدة) بعدين بناخد "الهيستوجرام" تبعها كميزة
        imgBright = manual_brightness(imgGray, 5)
        imgBlur = manual_blur(imgBright)
        imgSharp = manual_sharpen(imgBlur)
        histogram = get_hist_features(imgSharp)
        datasetEnhanced.append(histogram)

        lbl.append(cls)

# تحويل الليستات لمصفوفات عشان تعرف الmodels تتعامل معها
X_n = np.array(datasetNormal)
X_h = np.array(datasetEnhanced)
y = np.array(lbl)

#بنقسم الداتا لتدريب وتيست بنسبة 20%|80%
x_train_n, x_test_n, y_train, y_test = train_test_split(X_n, y, test_size=0.2, random_state=42)
x_train_h, x_test_h, _, _ = train_test_split(X_h, y, test_size=0.2, random_state=42)

# --- تدريب وتقييم الموديلات ---
# هون بنجرب 4 خوارزميات مشهورة ونقارن أداء كل وحدة على البكسلات العادية مقابل الهيستوجرام المحسن

print("\n" + "="*20 + " RESULTS " + "="*20)

# 1. تدريب KNN
knn_n = KNeighborsClassifier(n_neighbors=11).fit(x_train_n, y_train)
knn_h = KNeighborsClassifier(n_neighbors=11).fit(x_train_h, y_train)

# فحص الدقة للـ KNN
y_pred_knn_n = knn_n.predict(x_test_n)
print("\n=============== KNN: Normal Pixels ===============")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn_n):.2f}")
print(classification_report(y_test, y_pred_knn_n))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn_n))

y_pred_knn_h = knn_h.predict(x_test_h)
print("\n=============== KNN: Enhanced Hist ===============")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn_h):.2f}")
print(classification_report(y_test, y_pred_knn_h))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_knn_h))


# 2. تدريب Random Forest
rf_n = RandomForestClassifier(n_estimators=100, random_state=42).fit(x_train_n, y_train)
rf_h = RandomForestClassifier(n_estimators=100, random_state=42).fit(x_train_h, y_train)

# فحص الدقة للـ Random Forest
y_pred_rf_n = rf_n.predict(x_test_n)
print("\n=============== RF: Normal Pixels ===============")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_n):.2f}")
print(classification_report(y_test, y_pred_rf_n))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf_n))

y_pred_rf_h = rf_h.predict(x_test_h)
print("\n=============== RF: Enhanced Hist ===============")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_h):.2f}")
print(classification_report(y_test, y_pred_rf_h))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf_h))


# 3. تدريب Decision Tree
dt_n = DecisionTreeClassifier(max_depth=5, random_state=42).fit(x_train_n, y_train)
dt_h = DecisionTreeClassifier(max_depth=5, random_state=42).fit(x_train_h, y_train)

# فحص الدقة للـ Decision Tree
y_pred_dt_n = dt_n.predict(x_test_n)
print("\n=============== DT: Normal Pixels ===============")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt_n):.2f}")
print(classification_report(y_test, y_pred_dt_n))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt_n))

y_pred_dt_h = dt_h.predict(x_test_h)
print("\n=============== DT: Enhanced Hist ===============")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt_h):.2f}")
print(classification_report(y_test, y_pred_dt_h))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt_h))


# 4. تدريب SVM (Support Vector Machine)
svm_n = SVC(kernel='rbf', random_state=42).fit(x_train_n, y_train)
svm_h = SVC(kernel='rbf', random_state=42).fit(x_train_h, y_train)

# فحص الدقة للـ SVM
y_pred_svm_n = svm_n.predict(x_test_n)
print("\n=============== SVM: Normal Pixels ===============")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm_n):.2f}")
print(classification_report(y_test, y_pred_svm_n))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm_n))

y_pred_svm_h = svm_h.predict(x_test_h)
print("\n=============== SVM: Enhanced Hist ===============")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm_h):.2f}")
print(classification_report(y_test, y_pred_svm_h))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm_h))

# --- عرض النتائج ---
# في النهاية بنعرض الصورة الأصلية، وشكلها بعد ما نفذنا عليها فلاتر، والهيستوجرام اللي الموديل اتعلم منه
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(imgSharp, cmap='gray')
plt.title("Enhanced (After All Filters)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.bar(range(256), histogram, color='black', width=1.0)
plt.title("Manual Histogram Chart")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

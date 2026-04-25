import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# --- 1. تجهيز المجلدات والقوائم ---
lbl = []  # قائمة عشان نخزن فيها الأسماء (بس ولا كلب)
dataset = []  # قائمة للصور الأصلية قبل التعديل
dataset_enhanced = []  # قائمة للصور بعد ما نعدلها ونضبطها

classes = os.listdir("train")  # بنقرأ المجلدات اللي جوا train

# kernel sharpness (عشان نخلي الحواف واضحة وقوية)
sharpen_kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])


# دالة استخراج الهيستوغرام (بتحسب كم بكسل لكل درجة لون من 0 لـ 255)
def get_histogram(img):
    h, w = img.shape
    hist = np.zeros(256)  # بنعمل مصفوفة فاضية فيها 256 مكان

    for i in range(h):
        for j in range(w):
            c = img[i][j]  # بنشوف درجة اللون في هاي النقطة
            hist[c] = hist[c] + 1  # بنزيد العداد تبع هاي الدرجة

    return hist


# --- 2. تحميل الصور ومعالجتها ---
for i in classes:
    classPath = os.path.join("train", i)
    if not os.path.isdir(classPath): continue

    images_in_class = os.listdir(classPath)
    for j in images_in_class:
        imgPath = os.path.join(classPath, j)
        img_bgr = cv2.imread(imgPath)  # بنقرأ الصورة

        if img_bgr is None:
            continue

        # بنحول الصورة لرمادي (عشان الهيستوغرام أسهل) وبنوحد حجمها
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.resize(img_gray, (256, 256))

        # بنطلع الهيستوغرام للصورة الأصلية وبنخزنه
        hist_original = get_histogram(img_gray)
        dataset.append(hist_original)

        # --- مرحلة تحسين الصورة (Pipeline) ---
        # 1. بننظف الصورة من النويز (تغبيش خفيف)
        enhanced_img = cv2.GaussianBlur(img_gray, (3, 3), 0)
        # 2. بنزيد السطوع شوي عشان لو الصورة عتمة
        enhanced_img = cv2.convertScaleAbs(enhanced_img, beta=5)
        # 3. بنبين الحواف ونقوي التفاصيل
        enhanced_img = cv2.filter2D(enhanced_img, -1, sharpen_kernel)

        # بنطلع الهيستوغرام للصورة اللي "تضبطت" وبنخزنه
        hist_enhanced = get_histogram(enhanced_img)
        dataset_enhanced.append(hist_enhanced)

        lbl.append(i)  # بنخزن إنها بسة أو كلب

# بنحول القوائم لمصفوفات عشان الذكاء الاصطناعي يفهمها
lbl = np.array(lbl)
X_normal = np.array(dataset)
X_enhanced = np.array(dataset_enhanced)

# --- 3. تقسيم الشغل (تدريب وفحص) ---
# بنقسم الصور: 80% عشان الموديل يتعلم، و20% عشان نختبره ونشوف دقته
x_train_norm, x_test_norm, y_train, y_test = train_test_split(X_normal, lbl, test_size=0.2, random_state=42)
x_train_enh, x_test_enh, _, _ = train_test_split(X_enhanced, lbl, test_size=0.2, random_state=42)

# --- 4. تجربة KNN على الصور العادية ---
print("KNN Report: Histogram (Original)")
knn_orig = KNeighborsClassifier(n_neighbors=11)
knn_orig.fit(x_train_norm, y_train)
y_pred_knn_orig = knn_orig.predict(x_test_norm)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn_orig))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn_orig))

# --- 5. تجربة KNN على الصور المحسنة ---
print("KNN Report: Histogram (Enhanced)")
knn_enh = KNeighborsClassifier(n_neighbors=11)
knn_enh.fit(x_train_enh, y_train)
y_pred_knn_enh = knn_enh.predict(x_test_enh)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn_enh))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn_enh))

# --- 6. تجربة شجرة القرار (Tree) على الصور العادية ---
print("Tree Report: Histogram (Original)")
tree_orig = DecisionTreeClassifier(max_depth=10, random_state=42)
tree_orig.fit(x_train_norm, y_train)
y_pred_tree_orig = tree_orig.predict(x_test_norm)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree_orig))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tree_orig))

# --- 7. تجربة شجرة القرار (Tree) على الصور المحسنة ---
print("Tree Report: Histogram (Enhanced)")
tree_enh = DecisionTreeClassifier(max_depth=10, random_state=42)
tree_enh.fit(x_train_enh, y_train)
y_pred_tree_enh = tree_enh.predict(x_test_enh)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree_enh))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tree_enh))

# --- 8. رسم الصور عشان نشوف الفرق بعيننا ---
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('Original Gray Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(enhanced_img, cmap='gray')
plt.title('Enhanced Image (Bright + Blur + Sharp)')
plt.axis('off')
plt.show()

# --- 9. رسم الهيستوغرام عشان نشوف كيف توزيع الألوان تغير ---
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(hist_original)
plt.title('Original Histogram')

plt.subplot(1, 2, 2)
plt.plot(hist_enhanced)
plt.title('Enhanced Histogram')
plt.show()
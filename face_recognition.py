
"""Face_Recognition.ipynb


Original file is located at
    https://colab.research.google.com/drive/1UPs6Y7q7XXcrlPB1D9U6Z2INTXvyrgLJ
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

lfw = fetch_lfw_people(min_faces_per_person=70, resize=1, color=True, funneled=False, slice_=None)
images = lfw.images
labels = lfw.target
target_names = lfw.target_names
images_uint8 = (images * 255).astype(np.uint8) if images.max() <= 1.0 else images.astype(np.uint8)
images_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images_uint8]
selected_gray_images = images_gray[15:18]

images_denoised = [cv2.medianBlur(img, ksize=3) for img in selected_gray_images]

# Sharpening
kernel_sharpen = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
images_sharpened = [cv2.filter2D(img, -1, kernel_sharpen) for img in images_denoised]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
images_with_faces = []

for img in images_sharpened:
    img_faces = img.copy()
    faces = face_cascade.detectMultiScale(img_faces, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_faces, (x, y), (x+w, y+h), (255, 255, 255), 2)
    images_with_faces.append(img_faces)

plt.figure(figsize=(12, 6))
for i in range(3):
    plt.subplot(3, 4, i * 4 + 1)
    plt.imshow(selected_gray_images[i], cmap='gray')
    plt.title("Gray", fontsize=8)
    plt.axis('off')

    plt.subplot(3, 4, i * 4 + 2)
    plt.imshow(images_denoised[i], cmap='gray')
    plt.title("Median Blur", fontsize=8)
    plt.axis('off')

    plt.subplot(3, 4, i * 4 + 3)
    plt.imshow(images_sharpened[i], cmap='gray')
    plt.title("Sharpen", fontsize=8)
    plt.axis('off')

    plt.subplot(3, 4, i * 4 + 4)
    plt.imshow(images_with_faces[i], cmap='gray')
    plt.title("Face Detection", fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()

hog_features = []
hog_labels = []
image_size = (128, 128)

for i in range(len(images_gray)):
    img = images_gray[i]
    label = labels[i]
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face_resized = cv2.resize(face, image_size)
        features = hog(face_resized, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False)
        hog_features.append(features)
        hog_labels.append(label)
        break

hog_features = np.array(hog_features)
hog_labels = np.array(hog_labels)

X_train, X_test, y_train, y_test = train_test_split(
    hog_features, hog_labels, test_size=0.2, random_state=42, stratify=hog_labels)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = SVC(kernel='linear', probability=True)
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)


accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))


test_indices = [5, 12, 24]

for idx in test_indices:
    test_img_rgb = images_uint8[idx]
    test_img_gray = cv2.cvtColor(test_img_rgb, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(test_img_gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print(f"photo number {idx}: not find ")
        continue

    for (x, y, w, h) in faces:
        face = test_img_gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, image_size)

        test_features = hog(face_resized, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False)

        test_features_scaled = scaler.transform([test_features])
        predicted_label = clf.predict(test_features_scaled)[0]
        predicted_name = target_names[predicted_label]

        img_copy = test_img_rgb.copy()
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_copy, predicted_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
        plt.title(f"Image {idx} - Predicted: {predicted_name}")
        plt.axis('off')
        plt.show()
        break

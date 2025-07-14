
# 🧠 Face Recognition using HOG + SVM

This project demonstrates face recognition using the Labeled Faces in the Wild (LFW) dataset. It applies classical computer vision techniques:
- Grayscale conversion
- Image preprocessing (median filter + sharpening)
- Face detection using Haar Cascades
- Feature extraction using HOG (Histogram of Oriented Gradients)
- Classification using a linear SVM model

---

## 🔧 Libraries Used

- `cv2` (OpenCV): image processing & face detection
- `numpy`: numerical operations
- `matplotlib`: visualization
- `scikit-learn`: dataset, model training, evaluation
- `scikit-image`: HOG feature extraction

---

## 🚀 How the Code Works

### 1. **Load LFW Dataset**

```python
lfw = fetch_lfw_people(min_faces_per_person=70, resize=1, color=True)
```

- Loads colored face images (LFW) with at least 70 images per person.
- Provides: images, labels, and names.

---

### 2. **Image Preprocessing**

```python
images_gray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in images_uint8]
images_denoised = [cv2.medianBlur(img, 3) for img in selected_gray_images]
images_sharpened = [cv2.filter2D(img, -1, kernel_sharpen) for img in images_denoised]
```

- **Convert to grayscale**: simplifies feature extraction.
- **Median blur**: removes noise while preserving edges.
- **Sharpening**: enhances image details using a kernel.

---

### 3. **Face Detection**

```python
faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
```

- Detects faces using Haar cascades provided by OpenCV.
- Draws rectangles around detected faces for visualization.

---

### 4. **HOG Feature Extraction**

```python
features = hog(face_resized, orientations=9, pixels_per_cell=(8,8),
               cells_per_block=(3,3), block_norm='L2-Hys')
```

- Extracts robust features from each detected face.
- Uses HOG (Histogram of Oriented Gradients) – great for shape & texture.

---

### 5. **Train SVM Classifier**

```python
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train_scaled, y_train)
```

- Splits data into train/test sets.
- Scales features using `StandardScaler`.
- Trains a **Linear SVM** to classify the faces.

---

### 6. **Evaluate the Model**

```python
print(classification_report(y_test, y_pred, target_names=target_names))
```

- Prints precision, recall, and F1-score for each person in the test set.
- Shows model **accuracy** on unseen data.

---

### 7. **Test on Specific Images**

- Selects 3 test images.
- Detects faces → extracts HOG features → predicts the name.
- Visualizes the predictions with bounding boxes and labels.



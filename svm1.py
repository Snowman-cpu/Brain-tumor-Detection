import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set parameters
img_size = (64, 64)  # Reduced image size to save memory
batch_size = 200
train_dir = 'dataset/Training'
test_dir = 'dataset/Testing'

# Create file paths and labels
print("Preparing data paths...")
train_filepaths = []
train_labels = []
for fold in os.listdir(train_dir):
    fold_path = os.path.join(train_dir, fold)
    if os.path.isdir(fold_path):
        for file in os.listdir(fold_path):
            train_filepaths.append(os.path.join(fold_path, file))
            train_labels.append(fold)

test_filepaths = []
test_labels = []
for fold in os.listdir(test_dir):
    fold_path = os.path.join(test_dir, fold)
    if os.path.isdir(fold_path):
        for file in os.listdir(fold_path):
            test_filepaths.append(os.path.join(fold_path, file))
            test_labels.append(fold)

# Create dataframes
df_train = pd.DataFrame({'filepath': train_filepaths, 'label': train_labels})
df_test = pd.DataFrame({'filepath': test_filepaths, 'label': test_labels})

# Split test data into validation and test sets
df_valid, df_test = train_test_split(df_test, test_size=0.5, random_state=42)

print(f"Training samples: {len(df_train)}")
print(f"Validation samples: {len(df_valid)}")
print(f"Test samples: {len(df_test)}")

# Get class names
class_names = sorted(df_train['label'].unique())
print(f"Classes: {class_names}")

# Extract features in batches
def extract_features_batch(filepaths, img_size, start_idx, end_idx):
    batch_features = []
    
    for i in range(start_idx, min(end_idx, len(filepaths))):
        try:
            img = cv2.imread(filepaths[i])
            if img is not None:
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale to save memory
                batch_features.append(img.flatten())
        except Exception as e:
            print(f"Error processing {filepaths[i]}: {e}")
            continue
    
    return np.array(batch_features)

# Process training data in batches and prepare for SVM
print("Processing training data...")
X_train_all = []
for start_idx in range(0, len(df_train), batch_size):
    end_idx = start_idx + batch_size
    print(f"Processing batch {start_idx//batch_size + 1}/{(len(df_train)-1)//batch_size + 1}")
    
    X_batch = extract_features_batch(df_train['filepath'].values, img_size, start_idx, end_idx)
    if len(X_batch) > 0:
        X_train_all.append(X_batch)
    
    # Clear memory
    del X_batch
    gc.collect()

# Combine batches
X_train = np.vstack(X_train_all)
y_train = df_train['label'].values
del X_train_all
gc.collect()

# Apply scaling
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
del X_train
gc.collect()

# Apply PCA
print("Applying PCA...")
n_components = 100  # Adjust based on your memory constraints
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
print(f"Reduced dimensions to {X_train_pca.shape[1]} components")
del X_train_scaled
gc.collect()

# Train SVM model
print("Training SVM model...")
svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train_pca, y_train)
print("SVM model trained successfully")

# Process validation data
print("Processing validation data...")
X_valid_all = []
for start_idx in range(0, len(df_valid), batch_size):
    end_idx = start_idx + batch_size
    X_batch = extract_features_batch(df_valid['filepath'].values, img_size, start_idx, end_idx)
    if len(X_batch) > 0:
        X_valid_all.append(X_batch)
    
    del X_batch
    gc.collect()

X_valid = np.vstack(X_valid_all)
y_valid = df_valid['label'].values
del X_valid_all
gc.collect()

X_valid_scaled = scaler.transform(X_valid)
X_valid_pca = pca.transform(X_valid_scaled)
del X_valid
del X_valid_scaled
gc.collect()

# Evaluate on validation set
y_valid_pred = svm_model.predict(X_valid_pca)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
print(f"Validation Accuracy: {valid_accuracy:.4f}")
del X_valid_pca
gc.collect()

# Process test data
print("Processing test data...")
X_test_all = []
for start_idx in range(0, len(df_test), batch_size):
    end_idx = start_idx + batch_size
    X_batch = extract_features_batch(df_test['filepath'].values, img_size, start_idx, end_idx)
    if len(X_batch) > 0:
        X_test_all.append(X_batch)
    
    del X_batch
    gc.collect()

X_test = np.vstack(X_test_all)
y_test = df_test['label'].values
del X_test_all
gc.collect()

X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)
del X_test
del X_test_scaled
gc.collect()

# Evaluate on test set
y_test_pred = svm_model.predict(X_test_pca)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=class_names))

# Save model and results
model_data = {
    'model': svm_model,
    'scaler': scaler,
    'pca': pca,
    'class_names': class_names,
    'img_size': img_size
}

with open('svm_brain_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("SVM model saved successfully as 'svm_brain_model.pkl'")

# Save accuracy results to a text file
with open('svm_results.txt', 'w') as f:
    f.write(f"Validation Accuracy: {valid_accuracy:.4f}\n")
    f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_test_pred, target_names=class_names))

print("Results saved to 'svm_results.txt'")

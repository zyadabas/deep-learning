import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATA_DIR = r"D:\univ\Deep Learning\ass\archive_1\yawdd,cwe,glasses-dataset\train" 
CLASSES = ["Closed", "Open", "yawn", "no_yawn"] 
NUM_CLASSES = len(CLASSES)

IMG_SIZE = 64        
SEQ_LENGTH = 10      
STEP = 5             

X_data = []
y_data = []

print("Loading data...")

for label_idx, class_name in enumerate(CLASSES):
    folder_path = os.path.join(DATA_DIR, class_name)
    images = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: Directory {class_name} not found!")
        continue
        
    print(f"Processing directory: {class_name}...")
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  
            images.append(img)
            
    for i in range(0, len(images) - SEQ_LENGTH + 1, STEP):
        sequence = images[i : i + SEQ_LENGTH]
        X_data.append(sequence)
        y_data.append(label_idx)

X = np.array(X_data)
y = np.array(y_data)

y_cat = to_categorical(y, num_classes=NUM_CLASSES)

print("-" * 30)
print(f"Data formed successfully!")
print(f"Total sequences: {len(X)}")
print(f"X shape: {X.shape}") 
print("-" * 30)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

print("Building model...")
model = models.Sequential()

model.add(layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'), input_shape=(SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3)))
model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

model.add(layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')))
model.add(layers.TimeDistributed(layers.MaxPooling2D((2, 2))))

model.add(layers.TimeDistributed(layers.Flatten()))

model.add(layers.LSTM(64, return_sequences=False))

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print("Starting training...")
history = model.fit(
    X_train, y_train,
    epochs=15,          
    batch_size=16, 
    validation_data=(X_test, y_test)
)

model.save('drowsiness_4classes_model.h5')
print("Model trained and saved successfully.")
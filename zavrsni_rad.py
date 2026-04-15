import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix,
accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
# Parametri
IMG_SIZE = 224
BATCH_SIZE = 20
EPOCHS_NUMBER=40
TRAIN_DIR = 'C:/Users/ZenBook/Završni rad/prave_slike_podijeljene/train'
VAL_DIR = 'C:/Users/ZenBook/Završni rad/prave_slike_podijeljene/val'
TEST_DIR = 'C:/Users/ZenBook/Završni rad/prave_slike_podijeljene/test'
monocyte_count = len(os.listdir(os.path.join(TRAIN_DIR, 'monocyte')))
neutrophil_count = len(os.listdir(os.path.join(TRAIN_DIR, 'neutrophil')))
print(f"Monocyte slike: {monocyte_count}")
print(f"Neutrophil slike: {neutrophil_count}")
# Definicija ImageDataGenerator za treniranje s augmentacijom
train_datagen = ImageDataGenerator(
 rescale=1./255,
 rotation_range=10,
 width_shift_range=0.1,
 height_shift_range=0.1,
 shear_range=0.1,
 zoom_range=0.1,
 horizontal_flip=True,
 fill_mode='nearest'
)
# ImageDataGenerator za validaciju i testiranje (bez augmentacije)
test_datagen = ImageDataGenerator(rescale=1./255)
# Generatori podataka
train_generator = train_datagen.flow_from_directory(
 TRAIN_DIR,
 target_size=(IMG_SIZE, IMG_SIZE),
 batch_size=BATCH_SIZE,
 class_mode='binary'
)
print("Class indices:", train_generator.class_indices)
from sklearn.utils.class_weight import compute_class_weight
# Pretvaranje generatora u numpy array (za class_weight funkciju)
train_labels = train_generator.classes # ovo su 0 i 1
# Računanje težina
class_weights = compute_class_weight(
 class_weight='balanced',
 classes=np.unique(train_labels),
 y=train_labels
)
# Pretvaranje u dict (koji keras očekuje)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)
validation_generator = test_datagen.flow_from_directory(
 VAL_DIR,
 target_size=(IMG_SIZE, IMG_SIZE),
 batch_size=BATCH_SIZE,
 class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
 TEST_DIR,
 target_size=(IMG_SIZE, IMG_SIZE),
 batch_size=1, # bitno zbog prikaza pojedinačnih slika
 class_mode='binary',
 shuffle=True # da miješa slike
)
# Kreiranje modela
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Učitavanje MobileNetV2 s pretreniranom težinom
mobilenet = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE,
IMG_SIZE, 3),
 include_top=False,
weights='imagenet')
for layer in mobilenet.layers[:-30]: #zamrznuti svi osim zadnjih 30 slojeva -
fine tuning
 layer.trainable = False
# Kreiranje novog modela
model = models.Sequential([
 mobilenet,
 layers.GlobalAveragePooling2D(),
 layers.Dense(256, activation='relu'),
 layers.Dropout(0.4),
 layers.Dense(128, activation='relu'),
 layers.Dropout(0.3),
 layers.Dense(128, activation='relu'),
 layers.Dense(1, activation='sigmoid') # 2 klase: monocyte i neutrophil
])
# Kompajliranje modela s malim learning rate-om za fine-tuning

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
 loss='binary_crossentropy',
 metrics=['accuracy',
 tf.keras.metrics.Precision(name='precision'),
tf.keras.metrics.Recall(name='recall'),tf.keras.metrics.AUC(name='auc')])#area under the
curve (roc krivulja)
model.summary()
# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5,
restore_best_weights=True)
checkpoint = ModelCheckpoint("mobilenet_best_model.h5", save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
# Treniranje modela
epochs = EPOCHS_NUMBER
history = model.fit(
 train_generator,
 epochs=epochs,
 validation_data=validation_generator,
 callbacks=[early_stop, checkpoint],
 class_weight=class_weights
)
# Funkcija za crtanje grafova:
def plot_history(history):
 plt.plot(history.history['accuracy'], label='train accuracy')
 plt.plot(history.history['val_accuracy'], label='val accuracy')
 plt.title('Model Accuracy')
 plt.xlabel('Epoch')
 plt.ylabel('Accuracy')
 plt.legend()
 plt.show()
 plt.plot(history.history['loss'], label='train loss')
 plt.plot(history.history['val_loss'], label='val loss')
 plt.title('Model Loss')
 plt.xlabel('Epoch')
 plt.ylabel('Loss')
 plt.legend()
 plt.show()
plot_history(history)
# Predikcije
pred_probs = model.predict(test_generator)
y_pred = (pred_probs > 0.5).astype(int).flatten()
y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
# Evaluacija na testnim podacima(generiranih slika):
results = model.evaluate(test_generator)
for name, value in zip(model.metrics_names, results):
 print(f"{name}: {value:.4f}")
# Prikaz rezultata za 9 testnih (sintetičkih) slika:
class_names = list(test_generator.class_indices.keys())
plt.figure(figsize=(10, 10))
for i in range(9):img, label = test_generator[i]
 prediction = model.predict(img)
 predicted_class = class_names[1 if prediction[0] > 0.5 else 0]
 true_class = class_names[int(label[0])]

 correct = predicted_class == true_class
 color = 'green' if correct else 'red'

 plt.subplot(3, 3, i+1)
 plt.imshow(img[0])
 plt.axis('off')
 plt.title(f'Pred: {predicted_class}\nTrue: {true_class}', color=color)
plt.tight_layout()
plt.show()
# Metrike
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='macro')
rec = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
print("Točnost (Accuracy):", round(acc, 4))
print("Preciznost (Precision):", round(prec, 4))
print("Odziv (Recall):", round(rec, 4))
print("F1-mjera:", round(f1, 4))
# Matrica zabune
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels,
yticklabels=class_labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix - Stvarne slike')
plt.tight_layout()
plt.show()
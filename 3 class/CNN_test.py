import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization

combined_directory = 'E:\Downloads\detection_server_using_payload_TCP\data'
train_directory = r'E:\Downloads\detection_server_using_payload_TCP\model\train'
test_directory = r'E:\Downloads\detection_server_using_payload_TCP\model\test'


if not os.path.exists(train_directory):
    os.makedirs(train_directory)
if not os.path.exists(test_directory):
    os.makedirs(test_directory)


def split_data(source_dir, train_dir, test_dir, split_ratio=0.7):
    classes = os.listdir(source_dir)
    for cls in classes:
        cls_dir = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        images = os.listdir(cls_dir)
        num_images = len(images)
        num_train = int(num_images * split_ratio)

        np.random.shuffle(images)

        train_images = images[:num_train]
        test_images = images[num_train:]

        train_cls_dir = os.path.join(train_dir, cls)
        test_cls_dir = os.path.join(test_dir, cls)

        if not os.path.exists(train_cls_dir):
            os.makedirs(train_cls_dir)
        if not os.path.exists(test_cls_dir):
            os.makedirs(test_cls_dir)

        for img in train_images:
            shutil.copy(os.path.join(cls_dir, img), train_cls_dir)
        for img in test_images:
            shutil.copy(os.path.join(cls_dir, img), test_cls_dir)


split_data(combined_directory, train_directory, test_directory)


train_datagen = ImageDataGenerator(
    rescale=1./255,           
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    # rotation_range=20,  
    # width_shift_range=0.2, 
    # height_shift_range=0.2  
)

test_datagen = ImageDataGenerator(
    rescale=1./255 
)

train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(8, 8),     # Kích thước ảnh đầu vào
    batch_size=32,
    class_mode='categorical'  
)

test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=(8, 8),
    batch_size=32,
    class_mode='categorical'  
)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=(8, 8, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu', padding='same',),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  
])

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.001),
    # loss='binary_crossentropy',
    loss='categorical_crossentropy',

    metrics=['accuracy']
)


steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = test_generator.samples // test_generator.batch_size

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")


history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=30,  
    validation_data=test_generator,
    validation_steps=validation_steps
)

model.save('amazon_cnn_model.h5')

print("Đã huấn luyện và lưu mô hình CNN.")

def plot_history(history):
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Model Accuracy')

   
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Model Loss')
    
    plt.show()


plot_history(history)
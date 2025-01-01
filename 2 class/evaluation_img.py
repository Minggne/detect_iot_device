import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = load_model('amazon_cnn_model.h5')

def preprocess_image(image_path, target_size=(8, 8)):
    """Preprocess image: resize, normalize, and add batch dimension."""
    image = Image.open(image_path).resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = image_array / 255.0  # Normalize to [0, 1]
    return image_array

def predict_device_folder(device_folder, label):
    """Predict all images in a device folder and return true labels and predictions."""
    true_labels = []
    predictions = []
    image_files = [f for f in os.listdir(device_folder) if os.path.isfile(os.path.join(device_folder, f))]

    for image_file in image_files:
        image_path = os.path.join(device_folder, image_file)
        image_array = preprocess_image(image_path)
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        true_labels.append(label)
        predictions.append(predicted_class)

    return true_labels, predictions

def predict_category_folder(category_folder, label):
    """Predict all images in a category folder and return true labels and predictions."""
    true_labels = []
    predictions = []
    device_folders = [os.path.join(category_folder, d) for d in os.listdir(category_folder) if os.path.isdir(os.path.join(category_folder, d))]

    for device_folder in device_folders:
        device_labels, device_predictions = predict_device_folder(device_folder, label)
        true_labels.extend(device_labels)
        predictions.extend(device_predictions)

    return true_labels, predictions

def evaluate_model(amazon_dir, non_amazon_dir):
    """Evaluate model on both Amazon and non-Amazon directories."""
    # Predict for Amazon category (class 0)
    amazon_labels, amazon_predictions = predict_category_folder(amazon_dir, label=0)

    # Predict for Non-Amazon category (class 1)
    non_amazon_labels, non_amazon_predictions = predict_category_folder(non_amazon_dir, label=1)

    # Combine results
    y_true = np.array(amazon_labels + non_amazon_labels)
    y_pred = np.array(amazon_predictions + non_amazon_predictions)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Classification report with explicit labels
    report = classification_report(
        y_true, y_pred, target_names=['Amazon', 'Non_Amazon'], labels=[0, 1]
    )

    return cm, report
def plot_confusion_matrix(cm, class_names):
    """Plot the confusion matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# Define paths to category folders
amazon_folder = r'D:\detection_server_using_payload_TCP_2_class\danhgia_img\Amazon'
non_amazon_folder = r'D:\detection_server_using_payload_TCP_2_class\danhgia_img\non_Amazon'

# Run evaluation
confusion_mat, classification_rep = evaluate_model(amazon_folder, non_amazon_folder)

# Print results
print("Confusion Matrix:")
print(confusion_mat)
print("\nClassification Report:")
print(classification_rep)

plot_confusion_matrix(confusion_mat, class_names=['Amazon', 'Non_Amazon'])

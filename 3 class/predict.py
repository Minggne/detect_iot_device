import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


image_directory = r'E:\Downloads\detection_server_using_payload_TCP\danhgia_device\non_Amazon\Withings_2_tcp_features_non_Amazon'


model = load_model('amazon_cnn_model.h5')

def preprocess_image(image_path, target_size=(8, 8)):
    """Tiền xử lý ảnh: đổi kích thước, chuẩn hóa và thêm chiều batch."""
    # Đọc ảnh
    image = Image.open(image_path)
    
    image = image.resize(target_size)
   
    image_array = np.array(image)
   
    image_array = np.expand_dims(image_array, axis=0)
   
    image_array = image_array / 255.0
    return image_array

def predict_images(image_dir):
    """Dự đoán các ảnh trong thư mục và tổng hợp kết quả."""
    
    image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    
    predictions = []
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        image_array = preprocess_image(image_path)
        
        prediction = model.predict(image_array)
        
        predicted_class = np.argmax(prediction, axis=1)
        
        
        predictions.append(predicted_class[0])
    
    
    predictions = np.array(predictions)
    unique, counts = np.unique(predictions, return_counts=True)
    prediction_summary = dict(zip(unique, counts))
    
    return prediction_summary

def determine_device_class(prediction_summary):
    """Xác định lớp của thiết bị dựa trên kết quả dự đoán."""
    total_predictions = sum(prediction_summary.values())
    
    amazon_count = prediction_summary.get(0, 0)  
    google_count = prediction_summary.get(1, 0)
    other_count = prediction_summary.get(2, 0)  

    if amazon_count >= google_count and amazon_count >= other_count:
        return 'Amazon'
    elif google_count >= amazon_count and google_count >= other_count:
        return 'Google'
    else:
        return 'Other'


result = predict_images(image_directory)
device_class = determine_device_class(result)

print("Predicted results:", result)
print("Device class prediction:", device_class)

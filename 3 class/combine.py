import os
import shutil


root_source_directory = r'E:\Downloads\detection_server_using_payload_TCP\img\Other'
destination_directory = r'E:\Downloads\detection_server_using_payload_TCP\data\Other'


if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)


image_count = 0


for folder_name in os.listdir(root_source_directory):
    source_directory = os.path.join(root_source_directory, folder_name)
    
    
    if os.path.isdir(source_directory):
        for filename in os.listdir(source_directory):
            file_path = os.path.join(source_directory, filename)
            
            if os.path.isfile(file_path):
                
                new_filename = f'session_{image_count}.png'  
                destination_path = os.path.join(destination_directory, new_filename)
                
                shutil.copy(file_path, destination_path)
                image_count += 1

               
                if image_count >= 3000:
                    break
        if image_count >= 3000:
            break

print(f"Đã gộp tất cả các ảnh vào thư mục đích. Tổng số ảnh: {image_count}.")
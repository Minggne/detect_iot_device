import os
import pandas as pd
import numpy as np
from PIL import Image


def read_csv(file_path):
    return pd.read_csv(file_path)


def ip_to_int(ip):
    octets = ip.split('.')
    return sum(int(octet) << (8 * i) for i, octet in enumerate(reversed(octets)))


def payload_to_vector(payload, length=184):
    payload = payload[:length*2]  
    return [int(payload[i:i+2], 16) for i in range(0, len(payload), 2)]

def logarithmic_normalize(value, min_value, max_value):
    if value > 255:
        normalized_value = (np.log1p(value - min_value) / np.log1p(max_value - min_value)) * 255
        return int(normalized_value)
    else:
        return value


def normalize_feature_vector(vector, min_value, max_value):
    return [logarithmic_normalize(value, min_value, max_value) for value in vector]


def create_feature_vector(row):
    min_ip_value, max_ip_value = 0, 4294967295  
    min_port_value, max_port_value = 0, 65535  

    dest_ip = ip_to_int(row['Destination IP'])
    src_port = row['Source Port']
    dest_port = row['Destination Port']
    
    seq_num = row['seq_num']
    ack_num = row['ack_num']
    window_size = row['window_size']
    flags = int(row['flags'], 16)
    checksum = int(row['checksum'], 16)

   
    payload_vector = payload_to_vector(row['Payload'], length=184)

    
    feature_vector = [dest_ip, src_port, dest_port, seq_num, ack_num, window_size, flags, checksum] + payload_vector

    
    normalized_vector = normalize_feature_vector(feature_vector, 0, max(feature_vector))
    
    
    return normalized_vector[:192] + [0] * (192 - len(normalized_vector))


def vector_to_rgb_image(vector, image_size=(8, 8)):
    if len(vector) != 192:
        raise ValueError(f"Vector size must be 192 for an {image_size[0]}x{image_size[1]} RGB image. Current size: {len(vector)}")

    
    rgb_image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
    
    
    first_row = vector[:8]
    first_payload_byte = vector[8]
    rgb_image[0, 0] = [first_row[0], first_row[1], first_row[2]]  # R
    rgb_image[0, 1] = [first_row[3], first_row[4], first_row[5]]  # G
    rgb_image[0, 2] = [first_row[6], first_row[7], first_payload_byte]  # B

    
    payload_bytes = vector[9:192]
    row_index, col_index = 1, 0
    for byte in payload_bytes:
        rgb_image[row_index, col_index] = [byte, byte, byte]  # Gán giá trị cho RGB
        col_index += 1
        if col_index >= image_size[1]:
            col_index = 0
            row_index += 1
            if row_index >= image_size[0]:
                break
    
    
    image = Image.fromarray(rgb_image, mode='RGB')
    return image


def csv_to_images(file_path, output_folder):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = read_csv(file_path)

    for idx, row in df.iterrows():
        feature_vector = create_feature_vector(row)

        try:
            
            image = vector_to_rgb_image(feature_vector)
            image.save(f"{output_folder}/session_{idx}.png")
        except ValueError as e:
            print(f"Error creating image for index {idx}: {e}")


# csv_file_path = 'output_STA13.csv'


# base_name = os.path.splitext(os.path.basename(csv_file_path))[0]


# output_folder = f'output_images_{base_name}'


# csv_to_images(csv_file_path, output_folder)

csv_directory = r"E:\Downloads\detection_server_using_payload_TCP\ttttt\media_pcap_anonymized_csv"
output_base_folder = r"E:\Downloads\detection_server_using_payload_TCP\ttttt\img"

if not os.path.exists(output_base_folder):
    os.makedirs(output_base_folder)

for csv_file in os.listdir(csv_directory):
    if csv_file.endswith(".csv"):
        csv_file_path = os.path.join(csv_directory, csv_file)
        output_folder = os.path.join(output_base_folder, os.path.splitext(csv_file)[0])

        csv_to_images(csv_file_path, output_folder)
        print(f"Processed {csv_file} into {output_folder}")
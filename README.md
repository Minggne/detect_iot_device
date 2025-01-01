                              ***** DETECT IOT DEVICE *****
**** 2 class, 3 class cấu trúc đều như nhau. Dưới đây là hướng dẫn chạy project phân loại 2 class. ****


* 1. Dataset
** 1.1. Dữ liệu gốc từ Aalto, UNSW, Google (không cần đối với 2 class): https://drive.google.com/drive/folders/1sbERnssbCi4O-Tpu0s1hQE2N3p7il5Op?usp=drive_link
   1.2. Dữ liệu pcap trích ra để thực hiện mô hình: https://drive.google.com/drive/folders/1K-pzo1jY3ve4IiKrjpPt0c1_X6EnPhbV?usp=sharing
     - Sử dụng và lưu theo path: .\pcap\..
     
* 2. Trích xuất đặc trưng, chuyển đổi .pcap sang .csv
** 2.1. Code: collect.py
   2.2. Input, Output
     - Input: .\pcap\Amazon_pcap => Output: .\pcap2csv\Amazon_csv
     - Input: .\pcap\non_Amazon_pcap => Output: .\pcap2csv\non_Amazon.csv

* 3. Chuyển đổi file .csv sang .png
** 3.1. Code: convert.py
   3.2. Input, Output
     - Input: .\pcap2csv\Amazon.csv => Output: .\csv2png\Amazon
     - Input: .\pcap2csv\non_Amazon.csv => Output: .\csv2png\non_Amazon

* 4. Gộp tất cả các ảnh của mỗi class vào chung một folder để làm dữ liệu đầu vào cho mô hình
** 4.1. Code: combine.py
   4.2. Input, Output
     - Input: .\csv2png\Amazon => Output: .\data\Amazon
     - Input: .\csv2png\non_Amazon => Output: .\data\non_Amazon

* 5. Train model với thuật toán CNN
** 5.1. Code: CNN_test.py
   5.2. Input, Output
     - Input: .\data
     - Train directory: .\model\train
     - Test directory: .\model\test
     - Output: amazon_cnn_model.h5
    
* 6. Đánh giá predict
** 6.1. Với từng ảnh
     - Code: evaluation_img.py
     - Input: danhgia_img
** 6.2. Với thiết bị
     - Code: evaluation_device.py
     - Input: danhgia_device

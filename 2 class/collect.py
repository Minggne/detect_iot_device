import pyshark
import csv
import os


pcap_directory = r'D:\detection_server_using_payload_TCP_2_class\pcap\Amazon_pcap'
output_directory = r'D:\detection_server_using_payload_TCP_2_class\pcap2csv\Amazon_csv' 
max_packets = 350


if not os.path.exists(output_directory):
    os.makedirs(output_directory)


for pcap_file in os.listdir(pcap_directory):
    if pcap_file.endswith('.pcap'):  
        pcap_path = os.path.join(pcap_directory, pcap_file)
        output_csv = os.path.join(output_directory, pcap_file.replace('.pcap', '_tcp_features_Amazon.csv'))

       
        collected_data = []

        cap = pyshark.FileCapture(pcap_path, display_filter='tcp', tshark_path=r"D:\Wireshark\tshark.exe")

        count = 0
        has_non_80_port = False
        for i, packet in enumerate(cap):
            if count >= max_packets:
                break
            try:
                if hasattr(packet, 'tcp') and hasattr(packet.tcp, 'payload') and packet.tcp.payload:  
                    src_ip = packet.ip.src
                    dst_ip = packet.ip.dst
                    src_port = int(packet.tcp.srcport)
                    dst_port = int(packet.tcp.dstport)
                    seq_num = packet.tcp.seq
                    ack_num = packet.tcp.ack
                    window_size = packet.tcp.window_size_value
                    flags = packet.tcp.flags
                    checksum = packet.tcp.checksum

                    
                    payload = packet.tcp.payload.binary_value.hex()

                    if not dst_ip.startswith('192.168' and '10.10.10'):
                        
                        if dst_port != 80 or (dst_port == 80 and not has_non_80_port):
                           
                            collected_data.append([i+1, src_ip, dst_ip, src_port, dst_port, seq_num, ack_num, window_size,
                                                   flags, checksum, payload])
                            count += 1
                            has_non_80_port = True if dst_port != 80 else has_non_80_port

            except AttributeError:
                continue

        cap.close()

        
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            writer.writerow(['no.', 'Destination IP', 'Source Port', 'Destination Port', 'seq_num', 'ack_num',
                             'window_size', 'flags', 'checksum', 'Payload'])
            
            
            for row in collected_data:
                writer.writerow([row[0], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10]])

        print(f"Processed {pcap_file} -> {output_csv}")

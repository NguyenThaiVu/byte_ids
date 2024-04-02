import os
import sys
import numpy as np
import pandas as pd
import scapy
from scapy.all import wrpcap, Ether, rdpcap, PcapReader, Raw, IP, TCP, UDP
from scapy.utils import hexdump
import argparse
import multiprocessing 

from utils.read_file_utils import *


# Parse the arguments
parser = argparse.ArgumentParser(description='Description of your script')

parser.add_argument('--path_raw_folder_pcap', type=str, default=r"data/raw", help='Path to folder pcap containing raw dataset.')
parser.add_argument('--n_process', type=int, default=1, help='Number of cpus processing')
parser.add_argument('--max_length', type=int, default=100, help="Max length of header and payload byte sequence")
parser.add_argument('--path_out_processed_csv', type=str, default=r"data/processed", help='Path output folder of processed csv file')

args = parser.parse_args()

PATH_RAW_FOLDER_PCAP = args.path_raw_folder_pcap
N_process = args.n_process
MAX_LENGTH = args.max_length
PATH_OUT_PROCESSED_CSV_FOLDER = args.path_out_processed_csv
try:
    os.mkdir("data")  
    os.mkdir(PATH_OUT_PROCESSED_CSV_FOLDER)
except:  pass


def main():
    
    # Loop through all pcap folder [Adware, Benign, General_Malware]
    list_folder_pcap = os.listdir(PATH_RAW_FOLDER_PCAP)
    for folder_pcap in list_folder_pcap:
        path_folder_pcap = os.path.join(PATH_RAW_FOLDER_PCAP, folder_pcap)

        # Loop through all pcap file
        list_pcap_file = os.listdir(path_folder_pcap)
        input_args = [(os.path.join(path_folder_pcap, pcap_file_name), MAX_LENGTH) for pcap_file_name in list_pcap_file]
        print(f"[INFO] Label = {folder_pcap} | number of pcap file: {len(input_args)}")

        pool = multiprocessing.Pool(N_process)
        total_packet_bytes = pool.starmap(Get_List_Header_Payload_Bytes, input_args)
        pool.close()

        # Merge total packet bytes
        list_total_header_bytes = []
        list_total_payload_bytes = []
        list_label = []

        for (list_header_bytes, list_payload_bytes) in total_packet_bytes:
            list_total_header_bytes.extend(list_header_bytes)
            list_total_payload_bytes.extend(list_payload_bytes)
            list_label.extend([folder_pcap]*len(list_header_bytes))

        assert len(list_total_header_bytes) == len(list_total_payload_bytes) == len(list_label)
        print(f"Number of packets: {len(list_total_header_bytes)}")

        # Save to *.csv file
        df = pd.DataFrame({'header_bytes': list_total_header_bytes, 'payload_bytes': list_total_payload_bytes, 'labels': list_label})
        path_output_csv = os.path.join(PATH_OUT_PROCESSED_CSV_FOLDER, f"hex_data_{folder_pcap}.csv")
        df.to_csv(path_output_csv, index=False)

        print(f"Shape of df: {df.shape}")
        saved_file_size = os.path.getsize(path_output_csv)
        print(f"Size of saved file: {saved_file_size/2**30} GB\n")


main()
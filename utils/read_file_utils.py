import sys
sys.dont_write_bytecode = True
import os
import numpy as np
import pandas as pd
import multiprocessing 
import subprocess

import scapy
from scapy.all import wrpcap, Ether, rdpcap, PcapReader, Raw, IP, TCP, UDP, DNS


def makedir(path):
    try:
        os.mkdir(path)
    except Exception as E:
        pass


def Extract_Tracffic_Info_Packet(packet):
    """
    This function will extract the information of packet, including:
    <source ip, destination ip, source port, destination port, protocol>

    Parameters:
        packet (scapy object): indicate a single packet.

    Return:
        (src_ip, dst_ip, src_port, dst_port, protocol): tuple of 5 str.

    """

    if packet.haslayer('IP') == True:
        src_ip = packet['IP'].src
        dst_ip = packet['IP'].dst

        if packet.haslayer('TCP'):
            protocol = "TCP"
            src_port = packet['TCP'].sport
            dst_port = packet['TCP'].dport
        elif packet.haslayer('UDP'):
            protocol = "UDP"
            src_port = packet['UDP'].sport
            dst_port = packet['UDP'].dport
        else:
            return None
        
        return (src_ip, dst_ip, src_port, dst_port, protocol)
        
    else:
        return None 



def Calculate_Number_Packet(path_pcap_file):
    """
    This function will return the number of packet inside one pcap file (large pcap file)

    Parameters:
        path_pcap_file (str).

    Return:
        num_packets (int): number of packet inside one pcap file.
    """

    tshark_command = f"tshark -r {path_pcap_file} -Y 'frame.number' | wc -l"

    output = subprocess.check_output(tshark_command, shell=True, stderr=subprocess.STDOUT, universal_newlines=True)
    num_packets = int(output.strip())

    return num_packets


def Extract_Byte_From_Packet(packet, max_length=None):
    """
    This function extract the sequence bytes from scapy packet.

    Parameters:
        packet (scapy packet): packet read from scapy library.
        max_length (int): maximum length of byte sequence.

    Return:
        packet_byte_sequence (str): byte sequence of packet.
    """

    packet_byte_sequence = " ".join(f"{byte:02x}" for byte in bytes(packet))

    if max_length != None:
        packet_byte_sequence = packet_byte_sequence[:max_length*2 + (max_length-1)] 
    
    return packet_byte_sequence



def Extract_Header_Payload_From_Packet(packet, max_length=None):
    """
    This function extract the sequence bytes of header and sequece bytes of payload from scapy packet.

    Parameters:
        packet (scapy packet): packet read from scapy library.
        max_length (int): maximum length of byte sequence.

    Return:
        header_bytes (str): byte sequence of header.
        payload_bytes (str): byte sequence of payload.
    """

    # Get payload bytes (raw layer or DNS layer)
    if packet.haslayer(Raw):
        payload = packet.getlayer(Raw)[0]
        payload_bytes = " ".join(f"{byte:02x}" for byte in bytes(payload))
    elif packet.haslayer(DNS):
        payload = packet.getlayer(DNS)[0]
        payload_bytes = " ".join(f"{byte:02x}" for byte in bytes(payload))
    else:
        payload_bytes = ""

    # Get sequence byte of entire pcap file
    entire_packet = packet.getlayer(0)
    entire_packet_bytes = " ".join(f"{byte:02x}" for byte in bytes(entire_packet))

    # Get packet header 
    header_bytes = entire_packet_bytes[0:len(entire_packet_bytes) - len(payload_bytes)]

    if max_length != None:
        header_bytes = header_bytes[:max_length*2 + (max_length-1)] 
        payload_bytes = payload_bytes[:max_length*2 + (max_length-1)] 
    
    return (header_bytes, payload_bytes)



def Get_List_Header_Payload_Bytes(path_pcap_file, max_length=None):
    """
    This function read a large pcap file (contain a lot of packet). 
    Then return a list of two byte sequence, including byte sequence of HEADER and byte sequence of PAYLOAD.
    E.g.: [('e4 6f 13 e2', '16 8f 32 e2'), ('f8 08 00 45', '18 c3 a4 23'),...]

    NOTE:
    - We only load packet contains RAW layer or DNS layer.
    - If file size is too large, we only read 1_000_000 packet.

    Parameters:
        path_pcap_file (str): path to pcap file.
        max_length (int): the maximum length of byte sequence.

    Return:
        list_header_bytes (list): list of header's byte sequence.
        list_payload_bytes (list): list of payload's byte sequence.
    """

    list_header_bytes = []
    list_payload_bytes = []
    max_num_packet = None

    # If file size is too large, we only read 500_000 packet
    if os.path.getsize(path_pcap_file) > 100_000_000:  # 100MB
        max_num_packet = 500_000
        capture = PcapReader(path_pcap_file)
    else:
        capture = rdpcap(path_pcap_file)

    # Loop through all packet inside pcap file
    idx_packet = 0
    for packet in capture:
        if packet.haslayer(Raw) or packet.haslayer(DNS):
            (header_bytes, payload_bytes) = Extract_Header_Payload_From_Packet(packet, max_length)
            list_header_bytes.append(header_bytes)
            list_payload_bytes.append(payload_bytes)

            idx_packet += 1
        else:  pass

        if max_num_packet != None:
            if idx_packet > max_num_packet:  break

    return (list_header_bytes, list_payload_bytes)



def Calculate_Header_Payload_Length(packet):
    """
    This function calculate the length of the sequence header and payload bytes.

    Parameters:
        packet (scapy object).

    Return:
        length_header_bytes (int): the length of sequence HEADER bytes.
        length_payload_bytes (int): the length of sequence PAYLOAD bytes.
    """

    length_whole_packet_bytes = len(bytes(packet))

    if packet.haslayer(Raw):
        payload = packet.getlayer(Raw)[0]
    elif packet.haslayer(DNS):
        payload = packet.getlayer(DNS)[0]
    else:
        payload = ""

    length_payload_bytes = len(bytes(payload))

    length_header_bytes = length_whole_packet_bytes - length_payload_bytes
    
    return (length_header_bytes, length_payload_bytes)



def Calculate_Averge_Header_Payload_Length(path_pcap_file):
    """
    This function read a large pcap file (contain a lot of packet). 
    Then return the average sequence length of payload.

    Parameters:
        path_pcap_file (str): path to pcap file.

    Return:
        average_length_payload_bytes (int): average sequence length of payload.
    """

    list_length_header_bytes = []
    list_length_payload_bytes = []
    max_num_packet = None

    # If file size is too large, we only read 500_000 packet
    if os.path.getsize(path_pcap_file) > 500_000_000:
        max_num_packet = 500_000
        capture = PcapReader(path_pcap_file)
    else:
        capture = rdpcap(path_pcap_file)
    
    idx_packet = 0
    for packet in capture:
        if packet.haslayer(Raw) or packet.haslayer(DNS):
            (length_header_bytes, length_payload_bytes) = Calculate_Header_Payload_Length(packet)
            list_length_header_bytes.append(length_header_bytes)
            list_length_payload_bytes.append(length_payload_bytes)

            idx_packet += 1
        else:  pass

        if max_num_packet != None:
            if idx_packet > max_num_packet:  break

        
    return (list_length_header_bytes, list_length_payload_bytes)



"""
NOTE about packet and flow
- A network flow is a sequence of packets that share common attributes, 
    such as source and destination IP addresses, source and destination ports, and protocol type.
- In this experiment, if the flow contains more than 100 packets, we will split it into multiple flow, which has at most 100 packets.
"""

def Extract_Flows_From_Pcap_File(path_pcap_file, max_packets_per_flow=5):
    """
    This function read a large pcap file into list of packets. Then, it split those packets into flows.

    Parameters:
        path_pcap_file (str): path to pcap file.
        max_packets_per_flow (int): timeout value, i.e. each flow has at most 100 packets.

    Return:
        final_list_flows (list): list of flow, which is also a list containing multiple packets.
    """

    flows = {}   # Dictionary to store flows
    
    # Iterate over each packet
    packets = rdpcap(path_pcap_file)
    for packet in packets:
        try:
            # Extract information from packet
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            src_port = packet.sport
            dst_port = packet.dport
            protocol = packet[IP].proto
            
            # Define flow key based on packet characteristics
            flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
            
            # Add packet into flow
            if flow_key not in flows:  flows[flow_key] = [packet]
            else:  flows[flow_key].append(packet)
        except:
            pass

    list_flows = list(flows.values())

    # Split large flows into smaller flows with a maximum of max_packets_per_flow packets
    final_list_flows = []
    for flow in list_flows:
        if len(flow) <= max_packets_per_flow:
            final_list_flows.append(flow)
        else:
            # Split the flow into smaller flows with max_packets_per_flow packets each
            num_subflows = len(flow) // max_packets_per_flow
            for i in range(num_subflows):
                start_idx = i * max_packets_per_flow
                end_idx = (i + 1) * max_packets_per_flow
                final_list_flows.append(flow[start_idx:end_idx])

            final_list_flows.append(flow[end_idx:])
    
    return final_list_flows
# Multi-Modality Cross Attention Header and Payload for Intrusion Detection

We proposed the new method for intrusion detection, where we combine the information from both header and payload of packet network.

## 1. Overview

We inspect each individual network packet, which is represented as s sequence byte of header and sequence byte of payload.

## 2. Code information

- `config.py`: dataset path and hyper-parameters.
- `read_byte_data.py`: process pcap files to byte sequences (header and payload).
- `read_byte_data.sh`: execute `read_byte_data.py` file with some parameters.
- `train_byte_data.py`: training our model.
- `train_byte_data.sh`: execute `train_byte_data.py` file

- Folder `utils`: utility functions and helper modules:
    + `model_utils.py`: helper function to build model architecture.
    + `read_file_utils.py`: helper function to read dataset.
    + `train_utils.py`: helper function to train model.


## 3. Runing code

There are two main step to reproduce our method, including: **prepare dataset** and **training**

### 3.1. Prepare dataset
You need to prepare the folder, which have the following folder structure:
- data
    - raw
        - category 1
            - file1.pcap
            - file2.pcap
            - ...
        - category 2
            - file1.pcap
            - file2.pcap
            - ...
        - ...

**NOTE**: You can change some hyper-parameter of the **prepare dataset** step in file `read_byte_data.sh`. 

After finish, you will have the processed dataset in `data/processed` folder.

### 3.2. Training

In the second step, you can train your model by calling the file `train_byte_data.sh`.

**NOTE**: you can change some hyper-parameters in file `config.py`.


### 3.3. Download sample dataset
You can download the USTC-TF-2016 dataset from this [link](https://www.comet.com/thaiv7/artifacts/ustc_tf_2016).


## 4. Dataset information

| Dataset | Task | Number of categories |
|----------|----------|----------|
| USTC TFC 2016 | Classify different network flow | 19 |
| ISCX VPN NonVPN 2016 | Classify traffic network on VPN and nonVPN network | 7 |
| ISCX Tor 2016 | Classify traffic network on Tor Encryption network | 8 |
| CIC AAGM 2017 | Classify android malware | 3 |
| CIC-Bell-DNS 2021 | Classify benign or malicious (spam, phishing, and malware) domain name system | 4 |
| CIC IoT dataset 2022 | Classify behavior of IoT devices | 10 |


## 5. Performance

| Dataset | Accracy | F1 score |
|----------|----------|----------|
| USTC TFC 2016 | 99.30 | 95.58 |
| ISCX VPN NonVPN 2016 | 98.20 | 98.25 |
| ISCX Tor 2016 | 99.74 | 99.35 |
| CIC AAGM 2017 | 99.51 | 99.28 |
| CIC-Bell-DNS 2021 | 98.89 | 98.15 |
| CIC IoT dataset 2022 | 99.94 | 98.29 |
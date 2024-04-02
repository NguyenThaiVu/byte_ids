import os

# ---------------------------------------------------------------------------------------------
# Dataset hyper-parameters
# PATH_BYTE_FILE_FOLDER = r"data/processed"
PATH_BYTE_FILE_FOLDER = r"/home/necphy/ThaiVuNguyen/byte_ids/iscx_tor_2016/data/processed"

LIST_BYTE_FILE_NAME = os.listdir(PATH_BYTE_FILE_FOLDER)

MAX_HEADER_LENGTH = 50
MAX_PAYLOAD_LENGTH = 200

VOCAB_SIZE = 256
N_SAMPLES = None

TEST_SIZE = 0.2

# ---------------------------------------------------------------------------------------------
# Model hyper-parameters
num_layers = 2
d_model = 32
dff = 32
num_heads = 2
dropout_rate = 0.1

input_vocab_size = 256
output_classes = len(LIST_BYTE_FILE_NAME)

# ---------------------------------------------------------------------------------------------
# Model training hyper-parameters
BUFFER_SIZE = 20000
BATCH_SIZE = 512
EPOCHS = 2

OPTIMIZER = 'adam'

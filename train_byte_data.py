import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
tf.random.set_seed(42)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

from config import *
from utils.model_utils import *
from utils.train_utils import *



if __name__=="__main__": 

    """
    The source includes:
    1. Read dataset:
        - Read from csv.
        - Extract header and payload.

    2. Process text data:
        - Update token vocabulary.
        - Convert texts to sequences of integers.
        - Padding sequence of integers.

    3. Process label.

    4. Define model architecture

    5. Training model.

    6. Evaluation.
    """

    makedir("images")

    # ---------------------------------------------------------------------------------------------
    # 1. Read dataset

    # Read from csv.
    df = []
    for byte_file_name in LIST_BYTE_FILE_NAME:
        path_byte_file = os.path.join(PATH_BYTE_FILE_FOLDER, byte_file_name)
        sub_df = pd.read_csv(path_byte_file)
        df.append(sub_df)
    df = pd.concat(df, ignore_index=True)
    df = df.drop_duplicates(ignore_index=True).dropna()
    
    df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=42)
    print(f"Shape of df train: {df_train.shape}")
    print(f"Shape of df test: {df_test.shape}\n")

    # Extract header and payload
    (X_header_train, X_payload_train, y_train) = Extract_Header_Payload_Label(df_train)
    (X_header_test, X_payload_test, y_test) = Extract_Header_Payload_Label(df_test)

    # ---------------------------------------------------------------------------------------------
    # 2. Process text data
    
    # Update token vocabulary.
    tokenizer_header = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer_header.fit_on_texts(X_header_train)

    tokenizer_payload = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer_payload.fit_on_texts(X_payload_train)
    print("[INFO] Done update header and payload vocabulary.")

    # Transform text to sequence of integer
    X_header_train = tokenizer_header.texts_to_sequences(X_header_train)
    X_payload_train = tokenizer_payload.texts_to_sequences(X_payload_train)

    X_header_test = tokenizer_header.texts_to_sequences(X_header_test)
    X_payload_test = tokenizer_payload.texts_to_sequences(X_payload_test)
    print("[INFO] Done transform text to integers.")

    # Padding sequence of integers.
    X_header_train = pad_sequences(X_header_train, maxlen=MAX_HEADER_LENGTH, padding='post')
    X_payload_train = pad_sequences(X_payload_train, maxlen=MAX_PAYLOAD_LENGTH, padding='post')

    X_header_test = pad_sequences(X_header_test, maxlen=MAX_HEADER_LENGTH, padding='post')
    X_payload_test = pad_sequences(X_payload_test, maxlen=MAX_PAYLOAD_LENGTH, padding='post')
    print("[INFO] Done padding.\n")


    # ---------------------------------------------------------------------------------------------
    # 3. Process label
    lb = LabelBinarizer()
    lb.fit(y_train)

    y_train = lb.transform(y_train)

    y_test = lb.transform(y_test)
    y_test = np.argmax(y_test, axis=1)


    # ---------------------------------------------------------------------------------------------
    # 4. Define model architecture
    my_model = MyModel_v2(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size = input_vocab_size,
        output_classes = output_classes,
        dropout_rate=dropout_rate)
    

    header = np.random.randint(low = 0, high=input_vocab_size, size = (BATCH_SIZE, MAX_HEADER_LENGTH))
    payload = np.random.randint(low = 0, high=input_vocab_size, size = (BATCH_SIZE, MAX_PAYLOAD_LENGTH))

    print(f"[INFO] Shape of header: {header.shape}")
    print(f"[INFO] Shape of payload: {payload.shape}")

    output = my_model((header, payload))
    my_model.summary()


    # ---------------------------------------------------------------------------------------------
    # 5. Training model.
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-5, verbose=1)
    ]
    my_model.compile(optimizer=OPTIMIZER,  metrics=['accuracy'])

    history = my_model.fit([X_header_train, X_payload_train], y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, callbacks=callbacks)

    try: Plot_Acc_Loss_Training_Curve(history, file_name=os.path.join('images', "training_curve.png"))
    except: pass


    # ---------------------------------------------------------------------------------------------
    # 6. Evaluation.

    y_pred = my_model.predict([X_header_test, X_payload_test])
    y_pred = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print('Accuracy: %f' % accuracy)
    print('Precision: %f' % precision)
    print('Recall: %f' % recall)
    print('F1 score: %f' % f1)

    fig, ax = plt.subplots(figsize=(5,5))

    plt.rcParams.update({'font.size': 10})
    conf_matrix = confusion_matrix(y_test, y_pred)
    normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    normalized_conf_matrix = np.round(normalized_conf_matrix, 3)

    cm_display = ConfusionMatrixDisplay(confusion_matrix = normalized_conf_matrix, display_labels = lb.classes_)

    cm_display.plot(ax=ax, xticks_rotation = 'vertical', values_format=".4g")
    plt.savefig(os.path.join('images', "confusion_matrix.png"))
    # plt.savefig(os.path.join('images', "confusion_matrix.pdf"))

import os
import matplotlib.pyplot as plt


def makedir(path):
    try:
        os.mkdir(path)
    except Exception as E:
        pass


def Extract_Header_Payload_Label(df, header_col_name='header_bytes', payload_col_name='payload_bytes', label_col_name='labels'):
    """
    This function extract header, payload, label from dataframe.

    Parameters:
        df (pandas df): processed data, read from csv file.
        header_col_name (str)
        payload_col_name (str)
        label_col_name (str)

    Return:
        X_header (pandas series)
        X_payload (pandas series)
        y_label (pandas series)
    """
    X_header = df[header_col_name]
    X_payload = df[payload_col_name]
    y_label = df[label_col_name]

    return (X_header, X_payload, y_label)


def Plot_Acc_Loss_Training_Curve(history, file_name=None):
    plt.plot(history.history['accuracy'], 'o-')
    plt.plot(history.history['val_accuracy'], 'o-')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'], 'o-')
    plt.plot(history.history['val_loss'], 'o-')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show();

    if file_name != None:
        plt.savefig(file_name)

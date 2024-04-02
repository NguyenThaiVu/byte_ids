import os
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Flatten, Dense, Input, Conv1D, BatchNormalization, Activation, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import Mean
from tensorflow.keras import backend as K
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph



def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
  
  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)],axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)



# WE will use this to create a `PositionEmbedding` layer that looks-up a token's embedding vector and adds the position vector:
class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x
  

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-7)
    self.add = tf.keras.layers.Add()


class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)
   
    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x
  

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
  

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-7)

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x
  

class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate, name):
    super().__init__(name=name)

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
    
    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.
  

class MyModel_v2(tf.keras.Model):
    """
    The second version is difference from the first version, where we add the pyramid CONVOLUTION PROJECTION block.
    """

    def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, output_classes, dropout_rate=0.1):
        super().__init__()
        self.encoder_header = Encoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            vocab_size=input_vocab_size,
                            dropout_rate=dropout_rate, 
                            name='encoder_header')

        self.encoder_payload = Encoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            vocab_size=input_vocab_size,
                            dropout_rate=dropout_rate,
                            name='encoder_payload')

        self.cross_attention = CrossAttention(
                            num_heads=num_heads,
                            key_dim=d_model,
                            dropout=dropout_rate)
        
        self.final_layer = tf.keras.layers.Dense(output_classes, activation='softmax')
    
        self.loss_tracker = Mean(name="loss")
        self.acc_tracker = Mean(name="accuracy")

        self.conv_layer_1 = tf.keras.layers.Conv1D(filters=d_model, kernel_size=2, strides=2, padding="same")
        self.conv_layer_2 = tf.keras.layers.Conv1D(filters=d_model, kernel_size=2, strides=2, padding="same")


    def call(self, inputs, training=True):
        """
        This function define the forward pass of the model, where you specify input flows through the layers and get the output.

        NOTE: in inference mode (when training=False), we only output the predicted probability, instead of the learned embedding.

        Parameters:
            inputs (tensor): indicate the X (note: inputs does not contains y)
            training (boolean): boolean flag indicate whether model is called in training or inference mode.

        Return:
            output, header_encoding, payload_encoding (training model).
            output (inference).
        """

        header, payload  = inputs

        header_encoding = self.encoder_header(header)  # (batch_size, sequence_len, d_model)
        payload_encoding = self.encoder_payload(payload)

        # Pyramid network
        payload_encoding = self.conv_layer_1(payload_encoding)
        payload_encoding = self.conv_layer_2(payload_encoding)


        x = self.cross_attention(x=payload_encoding, context=header_encoding)

        x = Flatten()(x)

        output = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        if training == True:
            return output, header_encoding, payload_encoding
        else:
            return output

        
    def _compute_loss(self, y_true, y_pred, header_encoding, payload_encoding, alpha=0.1, margin=0.01):
        """
        This function will calculate the new custom loss function.

        Parameters:
            y_true (tensor): shape = (None, num_classes)
            y_pred (tensor): shape = (None, num_classes)
            header_encoding (tensor): output embedding of header, shape = (None, sequence_len, d_model)
            payload_encoding (tensor): output embedding of payload, shape = (None, sequence_len, d_model)

        Return:
            total_loss (tensor): shape = (None, 1)
        """

        crossentropy_loss = categorical_crossentropy(y_true, y_pred)

        encoding_distance = tf.math.reduce_mean(tf.math.square(header_encoding - payload_encoding))
        encoding_distance = tf.math.maximum([0.0], encoding_distance - margin)

        # encoding_distance = tf.math.reduce_sum(tf.math.square(header_encoding - payload_encoding))
        # encoding_distance = K.sum(K.square(header_encoding - payload_encoding), axis=-1)

        total_loss = crossentropy_loss + alpha * encoding_distance
        return total_loss


    def train_step(self, inputs):
        """
        This function overide the training step. 

        Parameters:
            inputs (tensor): we unpack it with (x, y)

        Return:
            dictionary: mapping metric names to current value.
        """

        (x, y) = inputs

        # GradientTape records every operation during the forward pass. 
        # We use it to compute the loss so we can get the gradients and apply them using the optimizer specified in `compile()`.
        with tf.GradientTape() as tape:
            y_pred, header_encoding, payload_encoding = self(x, training=True) # Forward pass

            loss = self._compute_loss(y, y_pred, header_encoding, payload_encoding) # Compute loss function

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Compute our own metrics
        self.loss_tracker.update_state(loss)
        accuracy = tf.keras.metrics.categorical_accuracy(y, y_pred)
        self.acc_tracker.update_state(accuracy)

        return {"loss": self.loss_tracker.result(), "accuracy": self.acc_tracker.result()}

    @property
    def metrics(self):
        # When we define metrics, the `reset_states()` can be called automatically at the start of each epoch.
        return [self.loss_tracker, self.acc_tracker]


    def test_step(self, inputs):
        """
        This function will help us evalute our model, i.e. when we call model.evaluate().
        
        NOTE: in this function, we will use the normal cross-entropy loss (instead of the custom loss - defined in train_test function).
            The reason is during evaluation step, we only want to evaluate the classification accuracy/loss instead of the learned embedding. 

        Parameters:
            inputs (tensor): we unpack it with (x, y)

        Return:
            dictionary: mapping metric names to current value.
        """

        (x, y) = inputs

        # Compute predictions
        y_pred = self(x, training=False)

        # Updates the metrics tracking the loss
        # loss = self._compute_loss(y_true=y, y_pred=y_pred)
        loss = categorical_crossentropy(y, y_pred)
        self.loss_tracker.update_state(loss)

        acc = tf.keras.metrics.categorical_accuracy(y, y_pred)
        self.acc_tracker.update_state(acc)

        return {'loss': loss, 'accuracy': acc}
    


class MyModel_Nonloss(tf.keras.Model):

    def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, output_classes, dropout_rate=0.1):
        super().__init__()
        self.encoder_header = Encoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            vocab_size=input_vocab_size,
                            dropout_rate=dropout_rate, 
                            name='encoder_header')

        self.encoder_payload = Encoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            vocab_size=input_vocab_size,
                            dropout_rate=dropout_rate,
                            name='encoder_payload')

        self.cross_attention = CrossAttention(
                            num_heads=num_heads,
                            key_dim=d_model,
                            dropout=dropout_rate)
        
        self.final_layer = tf.keras.layers.Dense(output_classes, activation='softmax')
    

    def call(self, inputs):
        """
        This function define the forward pass of the model, where you specify input flows through the layers and get the output.

        NOTE: in inference mode (when training=False), we only output the predicted probability, instead of the learned embedding.

        Parameters:
            inputs (tensor): indicate the X (note: inputs does not contains y)
            training (boolean): boolean flag indicate whether model is called in training or inference mode.

        Return:
            output, header_encoding, payload_encoding (training model).
            output (inference).
        """

        header, payload  = inputs

        header_encoding = self.encoder_header(header)  # (batch_size, sequence_len, d_model)
        payload_encoding = self.encoder_payload(payload)

        x = self.cross_attention(x=payload_encoding, context=header_encoding)

        x = Flatten()(x)

        output = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        return output


class MyModel_Non_CrossAttent(tf.keras.Model):

    def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, output_classes, dropout_rate=0.1):
        super().__init__()
        self.encoder_header = Encoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            vocab_size=input_vocab_size,
                            dropout_rate=dropout_rate, 
                            name='encoder_header')

        self.encoder_payload = Encoder(num_layers=num_layers, d_model=d_model,
                            num_heads=num_heads, dff=dff,
                            vocab_size=input_vocab_size,
                            dropout_rate=dropout_rate,
                            name='encoder_payload')
        
        self.merge_layer = tf.keras.layers.Concatenate()
        self.conv_layer_1 = tf.keras.layers.Conv1D(filters=d_model, kernel_size=2, strides=2, padding="same")


        self.final_layer = tf.keras.layers.Dense(output_classes, activation='softmax')
    

    def call(self, inputs):
        """
        This function define the forward pass of the model, where you specify input flows through the layers and get the output.

        NOTE: in inference mode (when training=False), we only output the predicted probability, instead of the learned embedding.

        Parameters:
            inputs (tensor): indicate the X (note: inputs does not contains y)
            training (boolean): boolean flag indicate whether model is called in training or inference mode.

        Return:
            output, header_encoding, payload_encoding (training model).
            output (inference).
        """

        header, payload  = inputs

        header_encoding = self.encoder_header(header)  # (batch_size, sequence_len, d_model)
        payload_encoding = self.encoder_payload(payload)

        # Conv projection block
        payload_encoding = self.conv_layer_1(payload_encoding)

        x = self.merge_layer([header_encoding, payload_encoding])

        x = Flatten()(x)

        output = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

        return output
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        
        self.dense = layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention = self.scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention)
        return output
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        
        self.mha = MultiHeadAttention(num_heads, d_model)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        
    def call(self, x, training, mask=None):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
        
    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                 maximum_position_encoding, rate=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_length = maximum_position_encoding
        
        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(maximum_position_encoding, d_model)
        
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, dff, rate)
            for _ in range(num_layers)
        ]
        
        self.dropout = layers.Dropout(rate)
        self.final_layer = layers.Dense(input_vocab_size)
        
    def call(self, inputs, *, training=False, mask=None):
        seq_len = tf.shape(inputs)[1]
        
        # Adding embedding and position encoding
        x = self.embedding(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.pos_encoding(x)
        
        x = self.dropout(x, training=training)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training, mask=mask)
            
        return self.final_layer(x)

def create_masks(seq):
    # Create look ahead mask
    # This mask ensures that for a given position in the sequence,
    # attention can only be paid to previous positions and the current position.
    seq_len = tf.shape(seq)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len, seq_len)), -1, 0
    )
    # The mask should be expanded to have the shape (batch_size, 1, seq_len, seq_len)
    # so it can be broadcasted to (batch_size, num_heads, seq_len, seq_len)
    # However, the MultiHeadAttention layer usually expects the mask for the attention scores directly,
    # which is (batch_size, num_heads, seq_len_q, seq_len_k).
    # Given Q, K, V are the same in self-attention, seq_len_q = seq_len_k = seq_len.
    # The mask from band_part is (seq_len, seq_len). We add batch and head dimensions if needed by the MHA layer,
    # but typically the MHA layer handles this if it receives a 2D or 3D mask.
    # For tf.keras.layers.MultiHeadAttention, a 2D mask (target_seq_len, source_seq_len) is often sufficient
    # and it will be broadcast. Let's return it as (1, seq_len, seq_len) for broadcasting to (batch_size, num_heads, ...)
    # Or simply (seq_len, seq_len) and let the MHA layer handle it if it supports it.
    # Given the current MHA adds `(mask * -1e9)`, it expects a mask that can be broadcast.
    # The original code had: padding_mask[:, tf.newaxis, tf.newaxis, :]
    # and then tf.maximum(padding_mask, look_ahead_mask)
    # look_ahead_mask is (seq_len, seq_len). For broadcasting to add with attention logits (batch, heads, seq_len, seq_len)
    # it needs to be (1, 1, seq_len, seq_len) or similar.
    # Let's keep it simple: the look_ahead_mask itself is what we need for causal attention.
    return look_ahead_mask # Shape: (seq_len, seq_len) 
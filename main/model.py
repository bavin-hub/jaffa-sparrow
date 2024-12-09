import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras_nlp
import tensorflow as tf
import keras
from keras import layers
from keras import ops
from keras.layers import TextVectorization
import numpy as np
import os
import string
import random
import tensorflow
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
from keras.initializers import random_normal
import json

# uncomment if you want to train with mixed-precision
# keras.mixed_precision.set_global_policy('mixed_float16')


def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = ops.arange(n_dest)[:, None]
    j = ops.arange(n_src)
    m = i >= j - n_src + n_dest
    mask = ops.cast(m, dtype)
    mask = ops.reshape(mask, [1, n_dest, n_src])
    mult = ops.concatenate(
        [ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])], 0
    )
    return ops.tile(mask, mult)

@keras.saving.register_keras_serializable()
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads, embed_dim, kernel_initializer=random_normal(stddev=0.02), dropout=0.1)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="gelu", kernel_initializer=random_normal(stddev=0.02)),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, "bool")
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim":self.embed_dim,
                "num_heads":self.num_heads,
                "ff_dim":self.ff_dim,
                "att":self.att,
                "ffn":self.ffn,
                "layernorm1":self.layernorm1,
                "layernorm2":self.layernorm2,
                "dropout1":self.dropout1,
                "dropout2":self.dropout2
            }
        )
        return config
    

@keras.saving.register_keras_serializable()
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__()
        # self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = keras_nlp.layers.ReversibleEmbedding(input_dim=vocab_size, output_dim=embed_dim, tie_weights=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(0, maxlen, 1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen":self.maxlen,
            "vocab_size":self.vocab_size,
            "embed_dim":self.embed_dim,
            "token_emb":self.token_emb,
            "pos_emb":self.pos_emb
        })
        return config
    



def create_model():

    with open("config.json", 'r') as file:
        hyperparameters = json.load(file)
    file.close()


    vocab_size = hyperparameters["vocab_size"]  
    maxlen = hyperparameters["seqlen"]
    embed_dim = hyperparameters["embed_dim"]
    num_heads = hyperparameters["num_heads"]
    feed_forward_dim = hyperparameters["feed_forward_dim"]
    num_layers = hyperparameters["num_layers"]

    inputs = layers.Input(shape=(None,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    for _ in range(num_layers):
        x  = TransformerBlock(embed_dim, num_heads, feed_forward_dim)(x)
    outputs = embedding_layer.token_emb(x, reverse=True)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])

    return model




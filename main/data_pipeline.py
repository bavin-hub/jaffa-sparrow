import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import sentencepiece
import matplotlib.pyplot as plt
import numpy as np
import json

def get_data():

    with open("config.json", "r") as file:
        hyperparameters = json.load(file)
    file.close()

    seqlen = hyperparameters["seqlen"] + 1
    bs = hyperparameters["batch_size"]
    buffer = hyperparameters["buffer_size"]
    
    # load tokenizer
    cwd = os.getcwd()
    tokenizer_dir = os.path.join(cwd, "tokenizer")
    tokenizer_model_file = os.path.join(tokenizer_dir, os.listdir(tokenizer_dir)[0])
    sp = sentencepiece.SentencePieceProcessor()
    sp.load(tokenizer_model_file)

    # load data
    data_dir = os.path.join(cwd,"data")
    data_path = os.path.join(data_dir, os.listdir(data_dir)[0])
    tokens = np.load(data_path)

    # create data pipeline
    data = tf.data.Dataset.from_tensor_slices(tokens)
    data = data.batch(seqlen, drop_remainder=True)

    def preprocess(token_sequence):
        inputs = token_sequence[:, :-1]
        outputs = token_sequence[:,1:]
        return (inputs, outputs)


    train_data = data.batch(batch_size=bs, drop_remainder=True)\
                     .map(preprocess)
                    #  .cache(tf.data.AUTOTUNE)\
                    #  .prefetch(tf.data.AUTOTUNE)

    train_subset_iterator = train_data.as_numpy_iterator()
    subset = train_subset_iterator.next()
    single_ex_x = tf.expand_dims(subset[0][0], axis=0)
    single_ex_y = tf.expand_dims(subset[0][0], axis=0)

    return subset[0], subset[1]

if __name__=="__main__":
    pass

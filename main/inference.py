import sentencepiece
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
import keras 
from keras import ops
from model import create_model
import json



def sample(logits):
        logits, indices = ops.top_k(logits, k=10,sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(ops.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

def inference(wts_path, model, maxlen, start_toks, cwd):
    
    tokenizer_dir = os.path.join(cwd, "tokenizer")
    tokenizer_model_file = os.path.join(tokenizer_dir, os.listdir(tokenizer_dir)[0])
    print("Tokenizer Model : ", tokenizer_model_file)

    print("Started Inference....")

    sp = sentencepiece.SentencePieceProcessor()
    sp.load(tokenizer_model_file)

    model.load_weights(wts_path)
    start = start_toks
    tokenized = sp.Encode(start)
    sample_idx = len(tokenized) - 1
    remaining = 256 - len(tokenized)
    tokenized_padded = tokenized + [0] * remaining
    inputs = tf.expand_dims(tokenized_padded, axis=0)

    ctr = maxlen
    while ctr:
        preds, _ = model.predict(inputs, verbose=0)
        sample_vector = preds[0][sample_idx]
        next_token = sample(sample_vector)
    
        tokenized.append(int(next_token))
        sample_idx = len(tokenized) - 1
        remaining = 256 - len(tokenized)
        tokenized_padded = tokenized + [0] * remaining
        inputs = tf.expand_dims(tokenized_padded, axis=0)

        ctr -= 1
    
    # print(tokenized)
    return sp.Decode(tokenized)

with open("config.json", 'r') as file:
    hyperparameters = json.load(file)
file.close()

generate_seqlen = hyperparameters["generate_seqlen"]

cwd = os.getcwd()
wts_dir = os.path.join(cwd, "checkpoint_weights")
wts_path = os.path.join(wts_dir, os.listdir(wts_dir)[0])
start_toks = "இந்தியாவின் ஆகாஷ்"
model_inference = create_model()
print(model_inference.summary())
model_inference.load_weights(wts_path)
print("Weights Loaded...")
res = inference(wts_path, model_inference, generate_seqlen, start_toks, cwd)
print("Inference Completed")
with open("inference_result.txt", "w") as file:
    file.write(res)
file.close()
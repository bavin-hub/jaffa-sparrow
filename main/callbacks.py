import keras
import numpy as np
from keras import ops
import sentencepiece
import json
import os


class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(
        self, max_tokens, start_tokens, index_to_word, top_k=10, print_every=1
    ):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.sp = index_to_word
        self.print_every = print_every
        self.k = top_k

        with open("config.json", 'r') as file:
            self.hyperparameters = json.load(file)
        file.close()

    def sample_from(self, logits):
        logits, indices = ops.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(ops.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, tokens):
        return self.sp.Decode(tokens)

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = self.hyperparameters["seqlen"] - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:self.hyperparameters["seqlen"]]
                sample_index = self.hyperparameters["seqlen"] - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x, verbose=0)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = self.detokenize([int(tok) for tok in self.start_tokens + tokens_generated])
        print(f"generated text:\n{txt}\n\n")


def model_checkpoints_callback(cwd):
    ckpts_dir = os.path.join(cwd, "checkpoint_weights")
    ckpts_file_path = os.path.join(ckpts_dir, "ckpts.weights.h5")
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=ckpts_file_path,
                                                                monitor='val_accuracy',
                                                                save_best_only=False,
                                                                save_weights_only=True,
                                                                save_freq='epoch')
    return model_checkpoint_callback

def get_callbacks():
    cwd = os.getcwd()

    tokenizer_dir = os.path.join(cwd, "tokenizer")
    tokenizer_model_file = os.path.join(tokenizer_dir, os.listdir(tokenizer_dir)[0])
    sp = sentencepiece.SentencePieceProcessor()
    sp.load(tokenizer_model_file)

    start_prompt = "இந்தியாவின் ஆகாஷ் சூப்பர்சானிக் ஏவுகணை வெற்றிகரமாக வானில்"
    # start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
    start_tokens = sp.Encode(start_prompt)
    num_tokens_generated = 200
    text_gen_callback = TextGenerator(num_tokens_generated, start_tokens, sp)

    model_checkpoints_cb = model_checkpoints_callback(cwd)

    return text_gen_callback, model_checkpoints_cb

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from model import create_model
from data_pipeline import get_data
from callbacks import get_callbacks
import json

# get the data
print("Preparing Data...")
data = get_data()

# get the callbacks
callbacks = get_callbacks()


# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():
  # Everything that creates variables should be under the strategy scope.
  # In general this is only model construction & `compile()`.
  model = create_model()
  model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer='adam')
  print("Model Initiated")
  

print(model.summary())


with open("config.json", "r") as file:
  hyperparameters = json.load(file)
file.close()

epochs = hyperparameters["epochs"]
batch_size = hyperparameters["batch_size"]

print("Training Started...")
history = model.fit(data[0], data[1], 
                    epochs=epochs,
                    shuffle=True, 
                    batch_size=64,
                    callbacks=callbacks)
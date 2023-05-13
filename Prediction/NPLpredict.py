import numpy as np
from keras.models import model_from_json
from keras_preprocessing.sequence import pad_sequences
import keras
from keras.datasets import imdb


# Load the model
file = open('model.json', 'r')
structure = file.read()
file.close()
loaded_model = model_from_json(structure)
loaded_model.load_weights(r"model.h5")
model = loaded_model

print(model.summary())

# encode function
word_index = imdb.get_word_index()
MAXLEN = 250
def encode_text(text):
  tokens = keras.preprocessing.text.text_to_word_sequence(text)
  tokens = [word_index[word] if word in word_index else 0 for word in tokens]
  return pad_sequences([tokens], MAXLEN)[0]

text = "I loved this movie, its was all good"
encoded = encode_text(text)

# prediction
pred = np.zeros((1,250))
pred[0] = encoded
result = model.predict(pred)

result = (result[0] * 100).item()
print(f'Positive: {result:.2f}%')

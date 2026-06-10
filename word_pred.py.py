"""
<a href="https://colab.research.google.com/github/Hars8430/New_word_prediction/blob/main/Untitled1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

# %%
import pandas as pd

# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
df=pd.read_csv('/content/qoute_dataset.csv')

# %%
df.head()


# %%
df['quote'][0]

# %%
df.shape

# %%
quotes=df['quote']

# %%
quotes.head()

# %%
quotes=quotes.str.lower()

# %%
import string
translator=str.maketrans('','',string.punctuation)
quotes=quotes.apply(lambda x:x.translate(translator))

# %%
quotes.head()

# %%
from tensorflow.keras.preprocessing.text import Tokenizer

# %%
vocab_size=9000
tokenizer=Tokenizer(num_words=vocab_size,oov_token='<oov>')
tokenizer.fit_on_texts(quotes)

# %%
word_index=tokenizer.word_index
print(len(word_index))
list(word_index.items())[:10]

# %%
sequence=tokenizer.texts_to_sequences(quotes)

# %%
for i in range (3):
  print(quotes[i])

# %%
for i in range (3):
  print (sequence[i])

# %%
X=[]
y=[]

# %%
for seq in sequence:
  for i in range(1,len(seq)):
    input_seq=seq[:i]
    output_seq=seq[i]
    X.append(input_seq)
    y.append(output_seq)

# %%
len(X)

# %%
max_len=max(len(x) for x in X)
print(max_len)

# %%
from tensorflow.keras.preprocessing.sequence import pad_sequences

# %%
X_padded=pad_sequences(X,maxlen=max_len,padding='pre')

# %%
X_padded

# %%
y=np.array(y)

# %%
X_padded.shape


# %%
y.shape

# %%
from tensorflow.keras.utils import to_categorical
y_one_hot=to_categorical (y,num_classes=vocab_size)

# %%
y_one_hot.shape

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,SimpleRNN

# %%
embedding_dim=50
rnn_units=128

# %%
rnn_model=Sequential()
rnn_model.add(
    Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=max_len)

)
rnn_model.add(
    SimpleRNN(units=rnn_units,activation='tanh')

)
rnn_model.add(
    Dense(units=vocab_size,activation='softmax')

)

# %%
rnn_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# %%
rnn_model.summary()

# %%
lstm_model= Sequential()
lstm_model.add(
    Embedding(input_dim=vocab_size,output_dim=embedding_dim,input_length=max_len))
lstm_model.add(LSTM(units=rnn_units,activation='tanh'))
lstm_model.add(Dense(units=vocab_size,activation='softmax'))

# %%
lstm_model.compile(
    optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy']
)

# %%
epochs=60
batch_size=128

# %%
history_rnn=rnn_model.fit(
    X_padded,y_one_hot,epochs=epochs,batch_size=batch_size,validation_split=0.1
)

# %%
lstm_model.save("lstm_model.h5")

# %%
index_to_word={}
for word,index in word_index.items():
  index_to_word[index]=word

# %%
from tensorflow.keras.preprocessing.sequence import pad_sequences

# %%
import tensorflow as tf

def predictor(model, tokenizer, text, max_len, temperature=1.0):
  text = text.lower()
  seq = tokenizer.texts_to_sequences([text])[0]
  seq = pad_sequences([seq], maxlen=max_len, padding='pre')
  pred = model.predict(seq, verbose=0)[0]
  pred = np.asarray(pred).astype('float64')
  pred = np.log(pred) / temperature
  exp_preds = np.exp(pred)
  pred = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, pred, 1)
  pred_index = np.argmax(probas)

  return index_to_word.get(pred_index, '')

# %%
seed_text="what is this room"
next_word=predictor(lstm_model,tokenizer,seed_text,max_len)
print(next_word)

# %%
def generate_text(model,tokenizer,seed_text,max_len,num_words,temperature=1.0):
  for _ in range(num_words):
    next_word=predictor(model,tokenizer,seed_text,max_len,temperature)
    if next_word== "":
      break
    seed_text+=" "+ next_word
  return seed_text

# %%
seed="what is wrong with"
generated_text=generate_text(lstm_model,tokenizer,seed,max_len,10, temperature=0.7)
print(generated_text)

# %%
lstm_model.save("lstm_model.h5")

# %%
import pickle
with open("tokenizer.pickle","wb") as handle:
  pickle.dump(tokenizer,handle)

# %%
with open("max_len.pickle","wb") as handle:
  pickle.dump(max_len,handle)

# %%


# %%
lstm_model.save("lstm_model.h5")


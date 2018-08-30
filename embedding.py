from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import numpy as np
text = open('Bible_for_word.txt','r').read().lower()
data = text.split( )
new_data = []
temp = []
target = []
vocab_size = len(list(set(data)))

for i,w in enumerate(data):
    if i==0: continue
    if(i%11==0):
        new_data.append(' '.join(temp))
        target.append(w)
        temp = []
    temp.append(w)
tk = Tokenizer()
tk.fit_on_texts(new_data)
train = tk.texts_to_sequences(new_data)
test = tk.texts_to_sequences(target)
padded_docs = pad_sequences(train, maxlen=10, padding='post')
y = to_categorical(test, num_classes=vocab_size)
model = Sequential()
model.add(Embedding(vocab_size,100,input_length = 10))
model.add(LSTM(100,activation='tanh'))
model.add(Dense(100,activation='tanh'))
model.add(Dense(vocab_size,activation='softmax'))
# model.add(Dense(1,activation = 'softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam')
sen_len = 10
reverse_word_map = dict(map(reversed, tk.word_index.items()))
def out(epoch,logs):
    test_sen = data[5:5+sen_len]
    senten = ' '.join(test_sen)
    x_test = array(tk.texts_to_sequences([test_sen])[0])
    x_test = x_test.reshape(1,10)
    pred_sen=' '
    for start in range(5,85):
        x_pred = model.predict(x_test,verbose=0)[0]
        pred_char = reverse_word_map[np.argmax(x_pred)]
        test_sen = test_sen[1:]+[pred_char]

        x_test = array(tk.texts_to_sequences([test_sen])[0])
        x_test = x_test.reshape(1,10)
        pred_sen+=pred_char+' '
    print(pred_sen)
from keras.callbacks import LambdaCallback
print_callback = LambdaCallback(on_epoch_end=out)
model.fit(padded_docs, y,batch_size = 256,epochs=60,callbacks=[print_callback])


test_sen = data[5:5+sen_len]
senten = ' '.join(test_sen)
x_test = array(array(tk.texts_to_sequences([test_sen])[0]))
pred_char = reverse_word_map[100]
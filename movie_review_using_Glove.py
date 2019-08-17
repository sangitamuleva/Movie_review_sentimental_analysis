import pandas as pd

train=pd.read_csv(r'G:\ml\ML_project\movie-review-sentiment-analysis-kernels-only\train.tsv',sep='\t')

test= pd.read_csv(r'G:\ml\ML_project\movie-review-sentiment-analysis-kernels-only\test.tsv',sep='\t')


from keras.utils import to_categorical

X = train['Phrase']  # get phrase as input

X_test=test['Phrase']

Y = to_categorical(train['Sentiment'].values)

full_text = list(train['Phrase'].values) + list(test['Phrase'].values)

from keras.preprocessing.text import Tokenizer

tk=Tokenizer()

tk.fit_on_texts(full_text)

train_token=tk.texts_to_sequences(X)
test_token=tk.texts_to_sequences(X_test)

from keras.preprocessing.sequence import pad_sequences

max_len = 200
X_train = pad_sequences(train_token, maxlen = max_len,padding='post')
X_testt = pad_sequences(test_token, maxlen = max_len,padding='post')


import numpy as np

embeding_index={}
f=open('glove.6B.300d.txt',encoding='utf8')
for line in f:
    values=line.split()
    word=values[0]
    coef=np.asarray(values[1:],dtype='float32')
    embeding_index[word]=coef

f.close()

vocab_size=len(tk.word_index)+1
print(vocab_size)
word_emb=np.zeros((vocab_size,300))

for word ,i in tk.word_index.items():
    # get vect of pre-trained model and assign to our model
    vect=embeding_index.get(word)

    if vect is not None:
        word_emb[i]=vect

from keras.models import Sequential
from keras.layers import Embedding,Flatten,Dense
# Now evalute model is before
model = Sequential()
model.add(Embedding(vocab_size,300,weights=[word_emb],input_length=200,trainable=False))
model.add(Flatten())
model.add(Dense(5,activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()

model.fit(X_train, Y, epochs=50, verbose=0)
# evaluate the model

loss, accuracy = model.evaluate(X_train, Y, verbose=0)
print('Accuracy: %f' % (accuracy*100))

model.save('best_model.h5')

from keras.models import load_model
model=load_model('best_model.h5')

y_pred=model.predict_classes(X_testt, verbose=1)

sub=pd.read_csv(r'G:\ml\ML_project\movie-review-sentiment-analysis-kernels-only\sub.csv')
sub.head(10)

y_test=sub['Sentiment']
print(y_test)

sub.Sentiment=y_pred
sub.to_csv('sampleSubmission.csv',index=False)
sub.head(10)

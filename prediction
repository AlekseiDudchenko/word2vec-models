from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, Conv2D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras import optimizers

from keras.layers import Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop

import pandas
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

#get data in words
some_data = pandas.read_csv(r"*.csv",
                            sep=' ; ', encoding = 'utf-8', engine='python', index_col=False)

labels = some_data.iloc[:,0]
samples = some_data.iloc[:,1]

num_of_diagnoses = len(set(labels))

#convert labels to categorical
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(labels)
encoded_labels = encoder.transform(labels)
Y_encoded = np_utils.to_categorical(encoded_labels)

import re 
import pymorphy2

def preprocessing_samples(samples):
    morph = pymorphy2.MorphAnalyzer()
    new_samples = []
    for s in samples:
        new_s = ""
        for w in s.split():
            r = re.compile("[^а-яА-Я ]+")
            w = r.sub('', w).lower()
            w = morph.parse(w)[0].normal_form  # делаем разбор
            new_s += w + " "
        new_samples.append(new_s)
    return new_samples

samples = preprocessing_samples(samples) # приводим все слова в выборках к леммам

#Word2Vec load
def words_in_sample(samples):
    """Max number of words in one sample"""
    max_len = 0
    for sample in samples:
        cur_sample = sample.split()
        max_len = len(cur_sample) if len(cur_sample) > max_len else max_len
    return max_len

max_words_in_sample = words_in_sample(samples)

#load word2vec model
from gensim.models import Word2Vec, KeyedVectors

# my model
#word2vecModel = Word2Vec.load("model_1.bin") #model_2.bin

# rusvectors model
#word2vecModel = KeyedVectors.load_word2vec_format(r"***.model") )

from collections import defaultdict
modelWord_from_word = dict()

for inx in range(len(word2vecModel.wv.vocab)):
    modelWord_from_word[word2vecModel.wv.index2word[inx].split('_')[0]] = word2vecModel.wv.index2word[inx]

from keras.preprocessing.text import text_to_word_sequence
def get_embedded_samples(samples, word2vecModel, words_in_sample):
    """get word2vec embeddings for given samples and words absent in given word2vec model"""
    new_x = np.zeros((len(samples), word2vecModel.vector_size*words_in_sample))
    absent_words = []
    i = 0 
    for sample in samples:
        current_sample = text_to_word_sequence(sample)
        newcur_x = np.zeros((1, word2vecModel.vector_size*max_words_in_sample))
        j = 0
        for word in current_sample:
            if word in modelWord_from_word:
                newcur_x[:, j:j+word2vecModel.vector_size] = (word2vecModel[modelWord_from_word[word]])
                j += word2vecModel.vector_size
            else:
                absent_words.append(word)
        new_x[i] = newcur_x
        i += 1
    return new_x, absent_words

new_x, absentWords = get_embedded_samples(samples, word2vecModel, max_words_in_sample)

def create_network():
    model_CNN = Sequential([
        Conv2D(32, (3, 3), activation='relu',
               input_shape=(max_words_in_sample,
                            word2vecModel.vector_size,
                            1),
               data_format="channels_last"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (2, 2), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_of_diagnoses, activation='softmax')])

    model_CNN.compile(optimizer=RMSprop(lr=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    return model_CNN

n_splits = 10
n_epochs = 35

x_CNN = new_x.reshape(new_x.shape[0], max_words_in_sample, word2vecModel.vector_size, 1)

kf = KFold(n_splits=n_splits, shuffle=True, random_state=2)
kf.get_n_splits(x_CNN)

f1_score_all = []
for train_index, test_index in kf.split(x_CNN):
    X_train, X_test = x_CNN[train_index], x_CNN[test_index]
    y_train, y_test = Y_encoded[train_index], Y_encoded[test_index]

    model_CNN = create_network()

    history_CNN = model_CNN.fit(X_train, y_train,
                                epochs=n_epochs,
                                verbose=1,
                                batch_size = 128,
                                validation_data=(X_test, y_test))

    # summarize history for accuracy
    plt.plot(history_CNN.history['acc'])
    plt.plot(history_CNN.history['val_acc'])
    plt.title('CNN accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')

    # evaluate
    print("acc ", model_CNN.evaluate(X_test, y_test, verbose = 0)[1])
    pred_cnn = model_CNN.predict_classes(X_test)
    metrica = f1_score(np.argmax(y_test,axis =1), pred_cnn, average='micro')
    print("F1 ", metrica)
    f1_score_all.append(metrica)

print(sum(f1_score_all)/len(f1_score_all))
plt.show()

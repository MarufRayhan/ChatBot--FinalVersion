from nltk.tokenize import word_tokenize as wt
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


class Utils:

    def __init__(self, load_dataset=None, words=None, token=None, df=None, max_length=0, encode=None, padded_doc=None,
                 output_one_hot=None):
        self.words = words
        self.token = token
        self.df = df
        self.max_length = max_length
        self.encode = encode
        self.load_dataset = load_dataset
        self.padded_doc = padded_doc
        self.output_one_hot = output_one_hot

    def cleaning(self, sentences: list):
        self.words = []
        for s in sentences:
            try:
                clean = str(re.sub("[^a-zA-Z0-9]", " ", s))
            except ImportError:
                clean = None
            w = wt(clean)
            # stemming
            self.words.append([i.lower() for i in w])
        return self.words

    def create_tokenizer(self, words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~'):
        self.token = Tokenizer(filters=filters)
        assert isinstance(words, object)
        self.token.fit_on_texts(words)
        return self.token

    def get_max_length(self, words):
        self.words = words
        print(max(self.words))
        return len(max(self.words, key=len))

    def encoding_doc(self, token, words):
        self.words = words
        assert isinstance(words, object)
        return token.texts_to_sequences(words)

    def padding_doc(self, encoded_doc, max_length):
        self.max_length = max_length
        return pad_sequences(encoded_doc, maxlen=self.max_length, padding="post")

    def one_hot(self, encode):
        self.encode = encode
        o = OneHotEncoder(sparse=False)
        return o.fit_transform(encode)

    def split_dataset(self, padded_doc, output_one_hot):
        self.padded_doc = padded_doc
        self.output_one_hot = output_one_hot
        train_x, val_x, train_y, val_y = train_test_split(self.padded_doc, self.output_one_hot, shuffle=True,
                                                          test_size=0.2)
        return train_x, val_x, train_y, val_y

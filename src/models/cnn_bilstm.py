from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout, Conv1D, MaxPooling1D
from src.models.base_model import Base


class CnnBilstm(Base):
    def __init__(self, vocab_size, max_length, embeddings_matrix):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embeddings_matrix = embeddings_matrix

    def setup(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size, 50, input_length=self.max_length, weights=[self.embeddings_matrix],
                            trainable=False))
        model.add(Dropout(.2))

        model.add(Conv1D(128, 5, activation="relu"))
        model.add(MaxPooling1D(pool_size=4))

        model.add(Bidirectional(LSTM(128)))
        model.add(Dropout(.5))

        model.add(Dense(66, activation="softmax"))

        self.base = model

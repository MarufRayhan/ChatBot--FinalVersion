from nltk.tokenize import word_tokenize
import re
import numpy as np
from src.utils.utils import Utils
from training.config import Config
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


class Train:

    def __init__(self, pred=None, model=None, train_x=None, train_y=None, val_x=None, val_y=None):
        self.pred = pred
        self.model = model
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y

    def train_model(self, model, train_x, train_y, val_x, val_y):
        self.model = model
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        try:
            self.model = load_model("model.h5")
            self.model.summary()
        except:
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
            model.summary()
            filename = 'model.h5'
            checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            hist = self.model.fit(self.train_x, self.train_y, epochs=Config.get_epochs(),
                                  batch_size=Config.get_batch_size(), validation_data=(self.val_x, self.val_y),
                                  callbacks=[checkpoint])
            print(hist)
            self.model = load_model("model.h5")

    def predictions(self, word_tokenizer, text, max_length, model):
        utils = Utils()
        clean = re.sub(r'[^ a-zA-Z0-9]', " ", text)
        test_word = word_tokenize(clean)
        test_word = [w.lower() for w in test_word]
        test_ls = word_tokenizer.texts_to_sequences(test_word)
        print(test_word)
        # Check for unknown words
        if [] in test_ls:
            test_ls = list(filter(None, test_ls))

        test_ls = np.array(test_ls).reshape(1, len(test_ls))
        print("test_ ls : ", test_ls)
        x = utils.padding_doc(test_ls, max_length)

        self.pred = model.predict_proba(x)

        return self.pred

    def get_final_output(self, pred, classes):
        self.pred = pred
        predictions = pred[0]
        classes = np.array(classes)
        ids = np.argsort(-predictions)
        print("ids : ", ids)
        classes = classes[ids]
        predictions = -np.sort(-predictions)

        for i in range(pred.shape[1]):
            print("%s has confidence = %s" % (classes[i], (predictions[i])))
        return classes[0], predictions[0]

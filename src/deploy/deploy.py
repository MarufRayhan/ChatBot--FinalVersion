from models.factory import get_network
from src.utils.utils import Utils
from src.data_preparation.data_loader import DataLoader
from training.config import Config
from training.train import Train
import numpy as np


def main():
    data_loader = DataLoader()
    load_dataset = data_loader.load_dataset(Config.get_dataset_path())
    print(load_dataset)
    utils = Utils(load_dataset)
    cleaned_words = utils.cleaning(load_dataset["sentences"])
    print("Cleaned Words : ", cleaned_words)
    word_tokenizer = utils.create_tokenizer(cleaned_words)
    vocab_size = len(word_tokenizer.word_index) + 1
    max_length = utils.get_max_length(cleaned_words)
    embeddings_matrix = data_loader.load_glove_embeddings(vocab_size, word_tokenizer)
    encoded_doc = utils.encoding_doc(word_tokenizer, cleaned_words)
    padded_doc = utils.padding_doc(encoded_doc, max_length)

    output_tokenizer = utils.create_tokenizer(load_dataset["unique_intent"], filters='!"#$%&()*+,-/:;<=>?@[\\]^`{|}~')

    encoded_output = utils.encoding_doc(output_tokenizer, load_dataset["intent"])
    encoded_output = np.array(encoded_output).reshape(len(encoded_output), 1)

    output_one_hot = utils.one_hot(encoded_output)
    train_x, val_x, train_y, val_y = utils.split_dataset(padded_doc, output_one_hot)

    model = get_network(Config.get_model_name(),
                        {"vocab_size": vocab_size, "max_length": max_length, "embeddings_matrix": embeddings_matrix})
    train = Train()
    params = {"model": model, "train_x": train_x, "train_y": train_y, "val_x": val_x, "val_y": val_y}
    train.train_model(**params)
    text = "office time?"
    pred = train.predictions(word_tokenizer, text, max_length, model)
    train.get_final_output(pred, load_dataset["unique_intent"])


if __name__ == "__main__":
    main()

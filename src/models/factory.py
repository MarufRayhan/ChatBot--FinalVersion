from models.cnn_bigru import CnnBigru
from src.models.cnn_lstm import CnnLstm
from src.models.cnn_bilstm import CnnBilstm


def get_network(name, params):
    """

    :param name: network name
    :param params: {key, value} parameters
    :return: keras network if it exists else raise KeyError
    """
    if name == 'CnnLstm':
        return CnnLstm(**params).get
    elif name == 'CnnBilstm':
        return CnnBilstm(**params).get
    elif name == 'CnnBigru':
        return  CnnBigru(**params).get
    else:
        raise KeyError("Unknown network " + name)

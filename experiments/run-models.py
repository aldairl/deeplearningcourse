import tensorflow as tf

from models import LogisticClasifier
from models import LSTM
from models import MLP

#datos
from data.EmojiDatasetWords import EmojiDatasetWords

#Parametros de la red

max_seq_len = 10
state_size = 280
learning_rate = 0.1
batch_size = 140
display_step = 10
validation_step = 50
model_path = None
restore_model = False

file_name = '..data/spanish_emojis-csv'
dataset = EmojiDatasetWords()
dataset.load_datasets()

def get_model(model_name):

    if model_name.lower() == 'logistic':
        return LogisticClasifier.LogisticClasifier

    if model_name.lower() == 'mlp':
        return MLP.MLP
    
    if model_name.lower() == 'lstm':
        return LSTM.LSTM

    return LSTM.LSTM


def run(args, model_name='logistic', report_result= True, epoc= 2000):
    dataset_info = dataset.get_train_test_val_data()

    train_data = dataset_info[0]
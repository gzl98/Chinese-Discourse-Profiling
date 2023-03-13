import os
import time

import torch

from src.utils import set_seed


class Config:
    def __init__(self, name=None, seed=0, speech=False):
        self.model_name = name
        self.train_time = time.strftime('%m-%d_%H.%M', time.localtime())
        self.num_layers = 1
        self.hidden_dim = 512
        self.bidirectional = True
        self.embedding_dim = 768
        self.dropout = 0.5
        self.bert_path = r'your_local_path/bert-base-chinese'
        self.bert_vocab_path = r'your_local_path/bert-base-chinese/vocab.txt'
        self.roberta_path = r'your_local_path/roberta-chinese'
        self.roberta_vocab_path = r'your_local_path/roberta-chinese/vocab.txt'
        self.epoch = 15
        self.require_improvement_epochs = 30
        self.seed = seed
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.speech = speech
        if speech:
            self.out_map = {'NA': 0, 'Speech': 1}
        else:
            self.out_map = {'NA': 0, 'Main': 1, 'Main_Consequence': 2, 'Cause_Specific': 3, 'Cause_General': 4,
                            'Distant_Historical': 5,
                            'Distant_Anecdotal': 6, 'Distant_Evaluation': 7, 'Distant_Expectations_Consequences': 8}
        self.out_dim = len(self.out_map)
        if self.model_name:
            self.work_path = './src/result/' + self.model_name + '/seed-' + str(self.seed)
            self.model_save_path = self.work_path + '/model'
            self.log_path = self.work_path + '/logs'
            if not os.path.isdir(self.model_save_path):
                os.makedirs(self.model_save_path)
            if not os.path.isdir(self.log_path):
                os.makedirs(self.log_path)

        set_seed(self.seed)

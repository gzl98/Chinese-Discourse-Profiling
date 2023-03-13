from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataloader import ChineseBertDataset
from src.original_model_bert import BertSentFeatClassifier
from src.trainer import BertTrainer


def parse_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--roberta', action='store_true', help='set to use roberta')
    args = parser.parse_args()
    return args


def train_chinese_bert(model, need_summary=False):
    model.init_weights()
    trainer = BertTrainer(config, model, train_dataloader, eval_dataloader, test_dataloader, need_summary=need_summary)
    trainer.train()


def evaluate_chinese_bert(model, is_test=False):
    model.load_state_dict(torch.load(config.model_save_path + '/model_' + config.train_time + '.ckpt'))
    trainer = BertTrainer(config, model, None, eval_dataloader, test_dataloader)
    trainer.evaluate(is_test=is_test, need_log=True)


if __name__ == '__main__':
    arguments = parse_arguments()
    config = Config(name='roberta_base_sent_feat', seed=arguments.seed)
    train_dataset = ChineseBertDataset(config, r'data/train/', roberta=arguments.roberta)
    eval_dataset = ChineseBertDataset(config, r'data/validation', roberta=arguments.roberta)
    test_dataset = ChineseBertDataset(config, r'data/test', roberta=arguments.roberta)
    train_dataloader = DataLoader(train_dataset, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False)
    test_dataloader = DataLoader(test_dataset, shuffle=False)
    model = BertSentFeatClassifier(config, roberta=arguments.roberta, bert_trainable=False).to(config.device)
    train_chinese_bert(model, need_summary=True)
    evaluate_chinese_bert(model, is_test=False)
    evaluate_chinese_bert(model, is_test=True)

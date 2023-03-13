import torch
import torch.nn as nn
from transformers import BertModel

from src.config import Config


class BertSentFeatClassifier(nn.Module):
    def __init__(self, config: Config, roberta=True, bert_trainable=False):
        super(BertSentFeatClassifier, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.roberta_path if roberta else config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = bert_trainable
        self.context_encoder = nn.LSTM(config.embedding_dim, config.hidden_dim, config.num_layers,
                                       batch_first=True, bidirectional=config.bidirectional)
        self.inner_pred = nn.Linear(config.hidden_dim * 2, config.embedding_dim)  # Prafulla 3 维度1024->1024
        self.pred = nn.Linear(config.hidden_dim * 2, config.out_dim)  # 1024->out_dim
        self.drop = nn.Dropout(config.dropout)
        self.ws1 = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)  # 1024->1024
        self.ws2 = nn.Linear(config.hidden_dim * 2, 1)  # 1024->1
        self.ws3 = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)  # 1024->1024
        self.ws4 = nn.Linear(config.hidden_dim * 2, 1)  # 1024->1
        self.softmax = nn.Softmax(dim=1)
        self.discourse_encoder = nn.LSTM(config.embedding_dim, config.hidden_dim, config.num_layers,
                                         batch_first=True, bidirectional=config.bidirectional)
        self.pre_pred = nn.Linear(config.hidden_dim * 2 * 5, config.hidden_dim * 2)  # 3072->1024

    def init_weights(self):
        init_list = [
            self.pred,
            self.ws1,
            self.ws2,
            self.inner_pred,
            self.ws3,
            self.ws4,
            self.pre_pred,
        ]
        for layer in init_list:
            nn.init.xavier_uniform_(layer.state_dict()['weight'])
            layer.bias.data.fill_(0)

        for name, param in self.context_encoder.state_dict().items():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        for name, param in self.discourse_encoder.state_dict().items():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def init_hidden(self, batch_size):
        return self.context_encoder.init_hidden(batch_size)

    def forward(self, token_ids, mask):
        token_ids = token_ids.squeeze(0)
        mask = mask.squeeze(0)
        embeddings = self.bert(token_ids, attention_mask=mask)
        outp_ctxt = embeddings[0]

        outp_2, _ = self.context_encoder(outp_ctxt)

        additional_tokens = torch.topk(outp_2[1:, :], 1, dim=1)
        additional_tokens = additional_tokens.values.reshape(additional_tokens.values.shape[0], -1)

        self_attention = torch.tanh(self.ws1(self.drop(outp_2)))
        self_attention = self.ws2(self.drop(self_attention)).squeeze()

        self_attention = self_attention + -10000 * (mask == 0).float()
        self_attention = self.softmax(self_attention)
        # word level attention
        sent_encoding = torch.sum(outp_2 * self_attention.unsqueeze(-1), dim=1)

        _inner_pred = torch.tanh(self.inner_pred(self.drop(sent_encoding)))
        inner_pred, _ = self.discourse_encoder(_inner_pred[None, :, :])

        self_attention = torch.tanh(self.ws3(self.drop(inner_pred)))
        self_attention = self.ws4(self.drop(self_attention)).squeeze(2)

        self_attention = self.softmax(self_attention)
        # sentence level attention
        disc_encoding = torch.sum(sent_encoding * self_attention.unsqueeze(-1), dim=1)
        inner_pred = inner_pred.squeeze()
        inner_pred = inner_pred[1:, :]

        disc_encoding = disc_encoding.expand(inner_pred.size())
        out_s = torch.cat([inner_pred,
                           disc_encoding * inner_pred,
                           disc_encoding - inner_pred,
                           additional_tokens,
                           sent_encoding[1:, :]],
                          1)
        pre_pred = torch.tanh(self.pre_pred(self.drop(out_s)))

        pred = self.pred(self.drop(pre_pred))
        return pred, outp_2, pre_pred, disc_encoding

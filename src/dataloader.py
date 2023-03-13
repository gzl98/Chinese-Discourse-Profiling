import os
from collections import OrderedDict

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


def sort_order(sids):
    temp = sorted([int(elem[1:]) for elem in sids])
    return ['S' + str(elem) for elem in temp]


class Document(object):

    def __init__(self, fname, domain):
        self.name = fname.split('/')[-1]
        self.domain = domain
        self.url = 'None'
        self.date = ['date']
        self.headline = ['headline']
        self.lead = []
        self.sentences = OrderedDict()
        self.tags = dict()
        self.sent_to_speech = None
        self.sent_to_event = None
        self.sids = []


def process_doc(file_name, domain):
    # Process text document
    f = open(file_name, 'r')
    doc = Document(file_name, domain)
    lead_para = False
    para_start = False
    para_index = 1
    sids = []
    for line in f:
        temp = line.strip()
        if temp == '':
            if para_start:
                para_index += 1
            if lead_para:
                for key in doc.sentences:
                    doc.lead += doc.sentences[key]
            lead_para = False
            continue

        temp = temp.split()
        if temp[0] == 'URL':
            doc.url = temp[1]
        elif temp[0] == 'DATE':
            doc.date = temp[1:]
        elif temp[0] == 'H':
            if len(temp[1:]) > 0:
                doc.headline = temp[1:]
        else:
            if temp[0] == 'S1':
                lead_para = True
            doc.sentences[temp[0]] = temp[1:]
            sids.append(temp[0])

    # Process annotation file
    f = open(file_name[:-3] + 'ann')
    sent_to_event = dict()
    sent_to_speech = dict()
    for line in f:
        temp = line.strip().split('\t')
        if len(temp) == 3:
            label = temp[1].split()[0]
            if label == 'Speech':
                sent_to_speech[temp[2]] = label
            else:
                # print(temp)
                sent_to_event[temp[2]] = label

    doc.sent_to_event = sent_to_event
    doc.sent_to_speech = sent_to_speech
    doc.sids = sort_order(sids)
    return doc


class ChineseBertDataset(Dataset):
    def __init__(self, config, data_path, roberta=True):
        super(ChineseBertDataset, self).__init__()
        self.config = config
        self.data_list = []
        self.tokenizer = BertTokenizer(config.roberta_vocab_path if roberta else config.bert_vocab_path)
        category_list = ['huanqiu', 'ckxx', 'xinhua', 'people']
        sub_category_list = ['finance', 'military', 'technology', 'politics']
        for category in category_list:
            category_path = os.path.join(data_path, category)
            for sub_category in sub_category_list:
                sub_category_path = os.path.join(category_path, sub_category)
                files = os.listdir(sub_category_path)
                for file in files:
                    if '.txt' in file:
                        doc = process_doc(os.path.join(sub_category_path, file), sub_category)
                        self.data_list.append(doc)

    def __getitem__(self, index):
        doc = self.data_list[index]
        sent, ls, out, sids = [], [], [], []
        sent.append(' '.join(doc.headline))
        ls.append(len(doc.headline))
        for sid in doc.sentences:
            if self.config.speech:
                out.append(self.config.out_map[doc.sent_to_speech.get(sid, 'NA')])
            else:
                out.append(self.config.out_map[doc.sent_to_event.get(sid)])
            sent.append(' '.join(doc.sentences[sid]))
            ls.append(len(doc.sentences[sid][0]))
            sids.append(sid)

        # tokenize
        # for sentence in sent:
        tokenized = self.tokenizer(sent, padding=True)
        token_ids = tokenized['input_ids']
        masks = tokenized['attention_mask']

        # 格式转换
        token_ids = torch.tensor(token_ids).to(self.config.device)
        masks = torch.tensor(masks).to(self.config.device)
        ls = torch.LongTensor(ls).to(self.config.device)
        out = torch.LongTensor(out).to(self.config.device)

        return token_ids, masks, ls, out

    def __len__(self):
        return len(self.data_list)

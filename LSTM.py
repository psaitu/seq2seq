import os
import time
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.utils.data.dataloader as dataloader
import torch.nn.functional as F
from ctcdecode import CTCBeamDecoder
import Levenshtein as L
import torch.optim as optim
import numpy as np
from tqdm import tqdm, tqdm_notebook
# mnist dataset
use_cuda = torch.cuda.is_available()

seed = 13
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device('cuda' if use_cuda else 'cpu')

train_phonemes = np.load('hw3p2-data-V2/wsj0_train.npy', encoding='latin1')
train_phonemes_labels = np.load('hw3p2-data-V2/wsj0_train_merged_labels.npy', encoding='latin1')
dev_phonemes = np.load('hw3p2-data-V2/wsj0_dev.npy', encoding='latin1')
dev_phonemes_labels = np.load('hw3p2-data-V2/wsj0_dev_merged_labels.npy', encoding='latin1')

PHONEME_LIST = [
    "0", "+BREATH+", "+COUGH+", "+NOISE+", "+SMACK+", "+UH+", "+UM+", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "SIL", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH"
]
PHONEME_MAP = [
    '0',  '_',   '+',   '~',   '!',   '-',   '@',   'a',   'A',   'h',   'o',   'w',   'y',   'b',   'c',   'd',   'D',   'e',   'r',   'E',   'f',   'g',   'H',   'i',   'I',   'j',   'k',   'l',   'm',   'n',   'G',   'O',   'Y',   'p',   'R',   's',   'S',   '.',   't',   'T',   'u',   'U',   'v',   'W',   '?',   'z',   'Z',  
]

train_phonemes_p1 = train_phonemes
train_phonemes_labels_p1 = train_phonemes_labels
dev_phonemes_p1 = dev_phonemes
dev_phonemes_labels_p1 = dev_phonemes_labels

total_train_batch_size = len(train_phonemes_p1)
total_train_label_batch_size = len(train_phonemes_labels_p1)
total_dev_batch_size = len(dev_phonemes_p1)
total_dev_label_batch_size = len(dev_phonemes_labels_p1)
pad_token = 0
feature_size = 40

train_lengths = [x.shape[0] for x in train_phonemes_p1]
train_label_lengths = [x.shape[0] for x in train_phonemes_labels_p1]
dev_lengths = [x.shape[0] for x in dev_phonemes_p1]
dev_label_lengths = [x.shape[0] for x in dev_phonemes_labels_p1]

def reorder_to_desc(lengths):
    return np.argsort(lengths)[::-1]

def all_lengths_max_lengths(inputs):
    lengths = [x.shape[0] for x in inputs]
    max_length = max(lengths)
    return lengths, max_length

def padd_only_features(unpadded_input, features_dim, pad_token):
    lengths, max_lengths = all_lengths_max_lengths(unpadded_input)
    
    total_train_batch_size = len(unpadded_input)
    feature_size = features_dim
    padded_input_holder = np.ones((total_train_batch_size, max_lengths, feature_size)) * pad_token
    
    for i, x_len in enumerate(lengths):
        padded_input_holder[i][0:x_len] = unpadded_input[i][:x_len]
        
    seq_order = reorder_to_desc(lengths)
    
    padded_input_list = [padded_input_holder[i] for i in seq_order]
    padded_input_sizes_list = [lengths[i] for i in seq_order]
    
    padded_input = torch.zeros((total_train_batch_size, max_lengths, feature_size))
    
    for i in range(len(padded_input)):
        padded_input[i] = torch.from_numpy(padded_input_list[i])
        
    return padded_input, padded_input_sizes_list
    

def padd_dataset(unpadded_input, features_dim, unpadded_labels, pad_token):
    feature_lengths, feautre_max_max_lengths = all_lengths_max_lengths(unpadded_input)
    label_lengths, laebl_max_lengths = all_lengths_max_lengths(unpadded_labels)
    
    total_train_batch_size = len(unpadded_input)
    feature_size = features_dim
    padded_input_holder = np.ones((total_train_batch_size, feautre_max_max_lengths, feature_size)) * pad_token
    
    for i, x_len in enumerate(feature_lengths):
        padded_input_holder[i][0:x_len] = unpadded_input[i][:x_len]
        
    seq_order = reorder_to_desc(feature_lengths)
    
    padded_input_list = [padded_input_holder[i] for i in seq_order]
    padded_input_sizes_list = [feature_lengths[i] for i in seq_order]
    
    padded_input = torch.zeros((total_train_batch_size, feautre_max_max_lengths, feature_size))
    
    for i in range(len(padded_input)):
        padded_input[i] = torch.from_numpy(padded_input_list[i])
        
    total_labels_batch_size = len(unpadded_labels)
    padded_label_holder = np.ones((total_labels_batch_size, laebl_max_lengths)) * pad_token
        
    for i, y_len in enumerate(label_lengths):
        padded_label_holder[i][0:y_len] = unpadded_labels[i][0:y_len]
    
    padded_label_list = [padded_label_holder[i] for i in seq_order]
    padded_label_sizes_list = [label_lengths[i] for i in seq_order]
    
    padded_labels = torch.zeros((total_labels_batch_size, laebl_max_lengths))
    
    for i in range(len(padded_labels)):
        padded_labels[i] = torch.from_numpy(padded_label_list[i])
        
    return padded_input, padded_input_sizes_list, padded_labels, padded_label_sizes_list


def collate(batch):
    inputs, input_lengths, targets, target_lengths = zip(*batch)
    padded_inputs, padded_input_lengths, padded_targets, padded_target_lengths = padd_dataset(inputs, feature_size, targets, 0)
    return padded_inputs, padded_input_lengths, padded_targets, padded_target_lengths

class SortedPhonemeDataSet(Dataset):
    def __init__(self, x, x_lens, y, y_lens):
        self.x = x
        self.x_lens = x_lens
        self.y = y
        self.y_lens = y_lens
        self.length = len(x)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.x[index], self.x_lens[index], self.y[index], self.y_lens[index]

class BiLSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super(BiLSTM, self).__init__(*args, **kwargs)
        
        self.stateh0 = nn.Parameter(torch.FloatTensor(2, 1, self.hidden_size).zero_())
        self.statec0 = nn.Parameter(torch.FloatTensor(2, 1, self.hidden_size).zero_())

    def forward(self, input):
        n = input.batch_sizes[0]
        return super(BiLSTM, self).forward(
            input,
            hx=(
                self.stateh0.expand(-1, n, -1).contiguous(),
                self.statec0.expand(-1, n, -1).contiguous()))



train_dataset = SortedPhonemeDataSet(train_phonemes, train_lengths, train_phonemes_labels, train_label_lengths)
test_dataset = SortedPhonemeDataSet(dev_phonemes, dev_lengths, dev_phonemes_labels, dev_label_lengths)

dataloader_args = dict(shuffle=False, batch_size=64, num_workers=8, pin_memory=True, collate_fn=collate) if use_cuda else dict(shuffle=False, batch_size=64, collate_fn=collate)
dataloader_args_dev = dict(shuffle=False, batch_size=79, num_workers=8, pin_memory=True, collate_fn=collate) if use_cuda else dict(shuffle=False, batch_size=79, collate_fn=collate)

train_loader = dataloader.DataLoader(train_dataset, **dataloader_args)
dev_loader = dataloader.DataLoader(test_dataset, **dataloader_args_dev)

def predict_sentence(model, x, x_lens):
    x, x_lens = Variable(x.cuda()), Variable(torch.tensor(x_lens))
    logits, _ = model(x, x_lens)
    ctcdecoder = CTCBeamDecoder(labels=PHONEME_MAP, beam_width=100, blank_id=0, log_probs_input=True)
    beam_result, beam_scores, timesteps, out_seq_len = ctcdecoder.decode(logits)
    decoded = []
    for i in range(beam_result.size(0)):
        chrs = ""
        if out_seq_len[i, 0] != 0:
            chrs = "".join(PHONEME_MAP[o] for o in beam_result[i, 0, :out_seq_len[i, 0]])
        decoded.append(chrs)
    return decoded


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.rnns = nn.ModuleList()
        self.rnns.append(BiLSTM(input_size=40, hidden_size=256))
        self.rnns.append(BiLSTM(input_size=256 * 2, hidden_size=256))
        self.rnns.append(BiLSTM(input_size=256 * 2, hidden_size=256))
        self.output_layer = nn.Linear(in_features=256 * 2, out_features=47)
        self.lsm = nn.LogSoftmax(dim=2)

    def forward(self, features, feature_lengths):
        h = rnn.pack_padded_sequence(features, feature_lengths.data.cpu().numpy(), batch_first=True)
        for l in self.rnns:
            h, _ = l(h)
        h, _ = rnn.pad_packed_sequence(h, batch_first=True)
        logits = self.output_layer(h)  # (t, n, l)
        logits = self.lsm(logits)
        return logits, feature_lengths

def true_sentences(labels):
    true_sentences = []
    for i in range(len(labels)):
        true_sentences.append("".join(PHONEME_MAP[l] for l in labels.type(torch.int)[i]))
    return true_sentences

def check_beam_results(model, loader):
    start = time.time()
    predictions = []
    true = []
    loss = 0
    for x, x_lens, y, y_lens in loader:
        predictions = predict_sentence(model, x, x_lens)
        true = true_sentences(y)
        loss += L.distance(predictions, true)
            
    ver = loss / 79
    
    torch.cuda.empty_cache()
    
    return ver

def train(epochs, model, criterion, optimizer, loader):
    best_eval = None
    for e in range(epochs):
        print("Training for epoch %d" % (e + 1))
        start = time.time()
        train_epoch_loss = 0
        val_epoch_loss = 0
        # train
        model.train()
        batch_idx = 0
        for x, x_lens, y, y_lens in loader:
            optimizer.zero_grad()
            x, x_lens, y, y_lens = Variable(x.cuda()), Variable(torch.tensor(x_lens)), Variable(y.cuda()), Variable(torch.tensor(y_lens))
            logits, _ = model(x, x_lens)
            logits_reshaped = logits.permute(1, 0, 2)
            loss = criterion(logits_reshaped, (y+1).type(torch.int32), x_lens, y_lens)
            loss.backward()
            
            optimizer.step()
            
            train_epoch_loss += loss.item()
            
            torch.cuda.empty_cache()
            del x
            del x_lens
            del y
            del y_lens
            del logits
            del _
            
            batch_idx += 1

        if e % 5 == 0:
            evaluate() 
            
        end = time.time()
        elapsed = end - start
        elapsed_string = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        print("Total Epoch Loss: %f" % train_epoch_loss)
        avg_loss = train_epoch_loss / 64
        print("Average Epoch Loss: %f" % avg_loss)
        print("Time Taken: %s" % elapsed_string)

def evaluate():
    model.eval()
    with torch.no_grad():
        avg_ldistance = check_beam_results(model, dev_loader)
    if best_eval is None or best_eval > avg_ldistance:
        best_eval = avg_ldistance
        torch.save(model.state_dict(), "models/best_model_yet.pth")
    print("Average L Distance: {}".format(avg_ldistance))


model = Model()
ctc = nn.CTCLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model.to(device)
train(20, model, ctc, optimizer, train_loader)

def collate_test(batch):
    inputs = batch
    padded_inputs, padded_input_lengths = padd_only_features(inputs, feature_size, 0)
    return padded_inputs, padded_input_lengths

class SortedTestPhonemeDataSet(Dataset):
    def __init__(self, x):
        self.x = x
        self.length = len(x)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.x[index]


test_data = np.load('hw3p2-data-V2/transformed_test_data.npy', encoding='latin1')
sub_set = SortedTestPhonemeDataSet(test_data)
dataloader_args_test = dict(shuffle=False, batch_size=1, num_workers=8, pin_memory=True, collate_fn=collate_test) if use_cuda else dict(shuffle=False, batch_size=1, collate_fn=collate_test)
sub_loader = dataloader.DataLoader(sub_set, **dataloader_args_test)

## Generate Submission File

model = Model()
model.load_state_dict(torch.load("models/best_model_yet.pth"))
model.to(device)
model.eval()

predicted = []
for index, (x, x_lens) in enumerate(sub_loader):
    predicted_sentence = predict_sentence(model, x, x_lens)
    predicted.append(predicted_sentence[0])

import pandas as pd
ids = []
for i in range(523):
    ids.append(i)

df = pd.DataFrame({'Id': ids,'Predicted': predicted})
df.to_csv('output_20.csv', index=False)
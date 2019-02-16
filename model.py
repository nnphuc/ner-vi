# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import nltk
import time
import torch
import torch.optim as optim
import torch.nn as nn
import os
import sys
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score
from torch.utils.data import Dataset

class WCDataset(Dataset):
   

    def __init__(self, wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths):
        
        self.wmaps = wmaps
        self.cmaps_f = cmaps_f
        self.cmaps_b = cmaps_b
        self.cmarkers_f = cmarkers_f
        self.cmarkers_b = cmarkers_b
        self.tmaps = tmaps
        self.wmap_lengths = wmap_lengths
        self.cmap_lengths = cmap_lengths

        self.data_size = self.wmaps.size(0)

    def __getitem__(self, i):
        return self.wmaps[i], self.cmaps_f[i], self.cmaps_b[i], self.cmarkers_f[i], self.cmarkers_b[i], self.tmaps[i], \
               self.wmap_lengths[i], self.cmap_lengths[i]

    def __len__(self):
        return self.data_size

class ViterbiDecoder():
    """
    Viterbi Decoder.
    """

    def __init__(self, tag_map):
      
        self.tagset_size = len(tag_map)
        self.start_tag = tag_map['<start>']
        self.end_tag = tag_map['<end>']

    def decode(self, scores, lengths):
        batch_size = scores.size(0)
        word_pad_len = scores.size(1)

        # Create a tensor to hold accumulated sequence scores at each current tag
        scores_upto_t = torch.zeros(batch_size, self.tagset_size)

        # Create a tensor to hold back-pointers
        # i.e., indices of the previous_tag that corresponds to maximum accumulated score at current tag
        # Let pads be the <end> tag index, since that was the last tag in the decoded sequence
        backpointers = torch.ones((batch_size, max(lengths), self.tagset_size), dtype=torch.long) * self.end_tag

        for t in range(max(lengths)):
            batch_size_t = sum([l > t for l in lengths])  # effective batch size (sans pads) at this timestep
            if t == 0:
                scores_upto_t[:batch_size_t] = scores[:batch_size_t, t, self.start_tag, :]  # (batch_size, tagset_size)
                backpointers[:batch_size_t, t, :] = torch.ones((batch_size_t, self.tagset_size),
                                                               dtype=torch.long) * self.start_tag
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and
                # choose the previous timestep that corresponds to the max. accumulated score for each current timestep
                scores_upto_t[:batch_size_t], backpointers[:batch_size_t, t, :] = torch.max(
                    scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1)  # (batch_size, tagset_size)

        # Decode/trace best path backwards
        decoded = torch.zeros((batch_size, backpointers.size(1)), dtype=torch.long)
        pointer = torch.ones((batch_size, 1),
                             dtype=torch.long) * self.end_tag  # the pointers at the ends are all <end> tags

        for t in list(reversed(range(backpointers.size(1)))):
            decoded[:, t] = torch.gather(backpointers[:, t, :], 1, pointer).squeeze(1)
            pointer = decoded[:, t].unsqueeze(1)  # (batch_size, 1)

        # Sanity check
        assert torch.equal(decoded[:, 0], torch.ones((batch_size), dtype=torch.long) * self.start_tag)

        # Remove the <starts> at the beginning, and append with <ends> (to compare to targets, if any)
        decoded = torch.cat([decoded[:, 1:], torch.ones((batch_size, 1), dtype=torch.long) * self.start_tag],
                            dim=1)

        return decoded

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Highway(nn.Module):
    """
    Highway Network.
    """

    def __init__(self, size, num_layers=1, dropout=0.5):
        super(Highway, self).__init__()
        self.size = size
        self.num_layers = num_layers
        self.transform = nn.ModuleList()  # list of transform layers
        self.gate = nn.ModuleList()  # list of gate layers
        self.dropout = nn.Dropout(p=dropout)

        for i in range(num_layers):
            transform = nn.Linear(size, size)
            gate = nn.Linear(size, size)
            self.transform.append(transform)
            self.gate.append(gate)

    def forward(self, x):
        transformed = nn.functional.relu(self.transform[0](x))  # transform input
        g = nn.functional.sigmoid(self.gate[0](x))  # calculate how much of the transformed input to keep

        out = g * transformed + (1 - g) * x  # combine input and transformed input in this ratio

        # If there are additional layers
        for i in range(1, self.num_layers):
            out = self.dropout(out)
            transformed = nn.functional.relu(self.transform[i](out))
            g = nn.functional.sigmoid(self.gate[i](out))

            out = g * transformed + (1 - g) * out

        return out


class CRF(nn.Module):
    """
    Conditional Random Field.
    """

    def __init__(self, hidden_dim, tagset_size):
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.emission = nn.Linear(hidden_dim, self.tagset_size)
        self.transition = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size))
        self.transition.data.zero_()

    def forward(self, feats):
        self.batch_size = feats.size(0)
        self.timesteps = feats.size(1)

        emission_scores = self.emission(feats)  # (batch_size, timesteps, tagset_size)
        emission_scores = emission_scores.unsqueeze(2).expand(self.batch_size, self.timesteps, self.tagset_size,
                                                              self.tagset_size)  # (batch_size, timesteps, tagset_size, tagset_size)

        crf_scores = emission_scores + self.transition.unsqueeze(0).unsqueeze(
            0)  # (batch_size, timesteps, tagset_size, tagset_size)
        return crf_scores


class LM_LSTM_CRF(nn.Module):

    def __init__(self, tagset_size, charset_size, char_emb_dim, char_rnn_dim, char_rnn_layers, vocab_size,
                 lm_vocab_size, word_emb_dim, word_rnn_dim, word_rnn_layers, dropout, highway_layers=1):

        super(LM_LSTM_CRF, self).__init__()

        self.tagset_size = tagset_size  # this is the size of the output vocab of the tagging model

        self.charset_size = charset_size
        self.char_emb_dim = char_emb_dim
        self.char_rnn_dim = char_rnn_dim
        self.char_rnn_layers = char_rnn_layers

        self.wordset_size = vocab_size  # this is the size of the input vocab (embedding layer) of the tagging model
        self.lm_vocab_size = lm_vocab_size  # this is the size of the output vocab of the language model
        self.word_emb_dim = word_emb_dim
        self.word_rnn_dim = word_rnn_dim
        self.word_rnn_layers = word_rnn_layers

        self.highway_layers = highway_layers

        self.dropout = nn.Dropout(p=dropout)

        self.char_embeds = nn.Embedding(self.charset_size, self.char_emb_dim)  # character embedding layer
        self.forw_char_lstm = nn.LSTM(self.char_emb_dim, self.char_rnn_dim, num_layers=self.char_rnn_layers,
                                      bidirectional=False, dropout=dropout)  # forward character LSTM
        self.back_char_lstm = nn.LSTM(self.char_emb_dim, self.char_rnn_dim, num_layers=self.char_rnn_layers,
                                      bidirectional=False, dropout=dropout)  # backward character LSTM

        self.word_embeds = nn.Embedding(self.wordset_size, self.word_emb_dim)  # word embedding layer
        self.word_blstm = nn.LSTM(self.word_emb_dim + self.char_rnn_dim * 2, self.word_rnn_dim // 2,
                                  num_layers=self.word_rnn_layers, bidirectional=True, dropout=dropout)  # word BLSTM

        self.crf = CRF((self.word_rnn_dim // 2) * 2, self.tagset_size)  # conditional random field

        self.forw_lm_hw = Highway(self.char_rnn_dim, num_layers=self.highway_layers,
                                  dropout=dropout)  # highway to transform forward char LSTM output for the forward language model
        self.back_lm_hw = Highway(self.char_rnn_dim, num_layers=self.highway_layers,
                                  dropout=dropout)  # highway to transform backward char LSTM output for the backward language model
        self.subword_hw = Highway(2 * self.char_rnn_dim, num_layers=self.highway_layers,
                                  dropout=dropout)  # highway to transform combined forward and backward char LSTM outputs for use in the word BLSTM

        self.forw_lm_out = nn.Linear(self.char_rnn_dim,
                                     self.lm_vocab_size)  # linear layer to find vocabulary scores for the forward language model
        self.back_lm_out = nn.Linear(self.char_rnn_dim,
                                     self.lm_vocab_size)  # linear layer to find vocabulary scores for the backward language model

    def init_word_embeddings(self, embeddings):
        self.word_embeds.weight = nn.Parameter(embeddings)

    def fine_tune_word_embeddings(self, fine_tune=False):
        for p in self.word_embeds.parameters():
            p.requires_grad = fine_tune

    def forward(self, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, wmaps, tmaps, wmap_lengths, cmap_lengths):
        self.batch_size = cmaps_f.size(0)
        self.word_pad_len = wmaps.size(1)

        # Sort by decreasing true char. sequence length
        cmap_lengths, char_sort_ind = cmap_lengths.sort(dim=0, descending=True)
        cmaps_f = cmaps_f[char_sort_ind]
        cmaps_b = cmaps_b[char_sort_ind]
        cmarkers_f = cmarkers_f[char_sort_ind]
        cmarkers_b = cmarkers_b[char_sort_ind]
        wmaps = wmaps[char_sort_ind]
        tmaps = tmaps[char_sort_ind]
        wmap_lengths = wmap_lengths[char_sort_ind]

        # Embedding look-up for characters
        cf = self.char_embeds(cmaps_f)  # (batch_size, char_pad_len, char_emb_dim)
        cb = self.char_embeds(cmaps_b)

        # Dropout
        cf = self.dropout(cf)  # (batch_size, char_pad_len, char_emb_dim)
        cb = self.dropout(cb)

        # Pack padded sequence
        cf = pack_padded_sequence(cf, cmap_lengths.tolist(),
                                  batch_first=True)  # packed sequence of char_emb_dim, with real sequence lengths
        cb = pack_padded_sequence(cb, cmap_lengths.tolist(), batch_first=True)

        # LSTM
        cf, _ = self.forw_char_lstm(cf)  # packed sequence of char_rnn_dim, with real sequence lengths
        cb, _ = self.back_char_lstm(cb)

        # Unpack packed sequence
        cf, _ = pad_packed_sequence(cf, batch_first=True)  # (batch_size, max_char_len_in_batch, char_rnn_dim)
        cb, _ = pad_packed_sequence(cb, batch_first=True)

        # Sanity check
        assert cf.size(1) == max(cmap_lengths.tolist()) == list(cmap_lengths)[0]

        # Select RNN outputs only at marker points (spaces in the character sequence)
        cmarkers_f = cmarkers_f.unsqueeze(2).expand(self.batch_size, self.word_pad_len, self.char_rnn_dim)
        cmarkers_b = cmarkers_b.unsqueeze(2).expand(self.batch_size, self.word_pad_len, self.char_rnn_dim)
        cf_selected = torch.gather(cf, 1, cmarkers_f)  # (batch_size, word_pad_len, char_rnn_dim)
        cb_selected = torch.gather(cb, 1, cmarkers_b)

        # Only for co-training, not useful for tagging after model is trained
        if self.training:
            lm_f = self.forw_lm_hw(self.dropout(cf_selected))  # (batch_size, word_pad_len, char_rnn_dim)
            lm_b = self.back_lm_hw(self.dropout(cb_selected))
            lm_f_scores = self.forw_lm_out(self.dropout(lm_f))  # (batch_size, word_pad_len, lm_vocab_size)
            lm_b_scores = self.back_lm_out(self.dropout(lm_b))

        # Sort by decreasing true word sequence length
        wmap_lengths, word_sort_ind = wmap_lengths.sort(dim=0, descending=True)
        wmaps = wmaps[word_sort_ind]
        tmaps = tmaps[word_sort_ind]
        cf_selected = cf_selected[word_sort_ind]  # for language model
        cb_selected = cb_selected[word_sort_ind]
        if self.training:
            lm_f_scores = lm_f_scores[word_sort_ind]
            lm_b_scores = lm_b_scores[word_sort_ind]

        # Embedding look-up for words
        w = self.word_embeds(wmaps)  # (batch_size, word_pad_len, word_emb_dim)
        w = self.dropout(w)

        # Sub-word information at each word
        subword = self.subword_hw(self.dropout(
            torch.cat((cf_selected, cb_selected), dim=2)))  # (batch_size, word_pad_len, 2 * char_rnn_dim)
        subword = self.dropout(subword)

        # Concatenate word embeddings and sub-word features
        w = torch.cat((w, subword), dim=2)  # (batch_size, word_pad_len, word_emb_dim + 2 * char_rnn_dim)

        # Pack padded sequence
        w = pack_padded_sequence(w, list(wmap_lengths),
                                 batch_first=True)  # packed sequence of word_emb_dim + 2 * char_rnn_dim, with real sequence lengths

        # LSTM
        w, _ = self.word_blstm(w)  # packed sequence of word_rnn_dim, with real sequence lengths

        # Unpack packed sequence
        w, _ = pad_packed_sequence(w, batch_first=True)  # (batch_size, max_word_len_in_batch, word_rnn_dim)
        w = self.dropout(w)

        crf_scores = self.crf(w)  # (batch_size, max_word_len_in_batch, tagset_size, tagset_size)

        if self.training:
            return crf_scores, lm_f_scores, lm_b_scores, wmaps, tmaps, wmap_lengths, word_sort_ind, char_sort_ind
        else:
            return crf_scores, wmaps, tmaps, wmap_lengths, word_sort_ind, char_sort_ind  # sort inds to reorder, if req.


class ViterbiLoss(nn.Module):
    """
    Viterbi Loss.
    """

    def __init__(self, tag_map):
        """
        :param tag_map: tag map
        """
        super(ViterbiLoss, self).__init__()
        self.tagset_size = len(tag_map)
        self.start_tag = tag_map['<start>']
        self.end_tag = tag_map['<end>']

    def forward(self, scores, targets, lengths):
    

        batch_size = scores.size(0)
        word_pad_len = scores.size(1)

        # Gold score

        targets = targets.unsqueeze(2)
        scores_at_targets = torch.gather(scores.view(batch_size, word_pad_len, -1), 2, targets).squeeze(
            2)  # (batch_size, word_pad_len)

        # Everything is already sorted by lengths
        scores_at_targets, _ = pack_padded_sequence(scores_at_targets, lengths, batch_first=True)
        gold_score = scores_at_targets.sum()

        # All paths' scores

        # Create a tensor to hold accumulated sequence scores at each current tag
        scores_upto_t = torch.zeros(batch_size, self.tagset_size).to(device)

        for t in range(max(lengths)):
            batch_size_t = sum([l > t for l in lengths])  # effective batch size (sans pads) at this timestep
            if t == 0:
                scores_upto_t[:batch_size_t] = scores[:batch_size_t, t, self.start_tag, :]  # (batch_size, tagset_size)
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and log-sum-exp
                # Remember, the cur_tag of the previous timestep is the prev_tag of this timestep
                # So, broadcast prev. timestep's cur_tag scores along cur. timestep's cur_tag dimension
                scores_upto_t[:batch_size_t] = log_sum_exp(
                    scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1)  # (batch_size, tagset_size)

        # We only need the final accumulated scores at the <end> tag
        all_paths_scores = scores_upto_t[:, self.end_tag].sum()

        viterbi_loss = all_paths_scores - gold_score
        viterbi_loss = viterbi_loss / batch_size

        return viterbi_loss

from collections import Counter
import codecs
import itertools
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
from torch.nn.utils.rnn import pack_padded_sequence


def read_words_tags(file, tag_ind, caseless=False):

    with codecs.open(file, 'r', 'utf-8') as f:
        lines = f.readlines()
    words = []
    tags = []
    temp_w = []
    temp_t = []
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            feats = line.rstrip('\n').split("\t")
            if len(feats)<2:
                continue
            feats[0] = feats[0].split(" ")
            feats[0] = "_".join(feats[0])
            temp_w.append(feats[0].lower() if caseless else feats[0])
            temp_t.append(feats[tag_ind])
        elif len(temp_w) > 0:
            assert len(temp_w) == len(temp_t)
            words.append(temp_w)
            tags.append(temp_t)
            temp_w = []
            temp_t = []
    # last sentence
    if len(temp_w) > 0:
        assert len(temp_w) == len(temp_t)
        words.append(temp_w)
        tags.append(temp_t)

    # Sanity check
    assert len(words) == len(tags)

    return words, tags


def create_maps(words, tags, min_word_freq=2, min_char_freq=1):
    
    word_freq = Counter()
    char_freq = Counter()
    tag_map = set()
    for w, t in zip(words, tags):
        word_freq.update(w)
        char_freq.update(list(reduce(lambda x, y: list(x) + [' '] + list(y), w)))
        tag_map.update(t)

    word_map = {k.lower(): v + 1 for v, k in enumerate([w for w in word_freq.keys() if word_freq[w] > min_word_freq])}
    char_map = {k: v + 1 for v, k in enumerate([c for c in char_freq.keys() if char_freq[c] > min_char_freq])}
    tag_map = {k: v + 1 for v, k in enumerate(tag_map)}

    word_map['<pad>'] = 0
    word_map['<end>'] = len(word_map)
    word_map['<unk>'] = len(word_map)
    char_map['<pad>'] = 0
    char_map['<end>'] = len(char_map)
    char_map['<unk>'] = len(char_map)
    tag_map['<pad>'] = 0
    tag_map['<start>'] = len(tag_map)
    tag_map['<end>'] = len(tag_map)

    return word_map, char_map, tag_map


def create_input_tensors(words, tags, word_map, char_map, tag_map):
    
    wmaps = list(map(lambda s: list(map(lambda w: word_map.get(w.lower(), word_map['<unk>']), s)) + [word_map['<end>']], words))

    chars_f = list(map(lambda s: list(reduce(lambda x, y: list(x) + [' '] + list(y), s)) + [' '], words))
   
    chars_b = list(
        map(lambda s: list(reversed([' '] + list(reduce(lambda x, y: list(x) + [' '] + list(y), s)))), words))

    cmaps_f = list(
        map(lambda s: list(map(lambda c: char_map.get(c, char_map['<unk>']), s)) + [char_map['<end>']], chars_f))
    
    cmaps_b = list(
        map(lambda s: list(map(lambda c: char_map.get(c, char_map['<unk>']), s)) + [char_map['<end>']], chars_b))

   
    cmarkers_f = list(map(lambda s: [ind for ind in range(len(s)) if s[ind] == char_map[' ']] + [len(s) - 1], cmaps_f))
  
    cmarkers_b = list(
        map(lambda s: list(reversed([ind for ind in range(len(s)) if s[ind] == char_map[' ']])) + [len(s) - 1],
            cmaps_b))

  
    tmaps = list(map(lambda s: list(map(lambda t: tag_map[t], s)) + [tag_map['<end>']], tags))
  
    tmaps = list(map(lambda s: [tag_map['<start>'] * len(tag_map) + s[0]] + [s[i - 1] * len(tag_map) + s[i] for i in
                                                                             range(1, len(s))], tmaps))
   
    word_pad_len = max(list(map(lambda s: len(s), wmaps)))
    char_pad_len = max(list(map(lambda s: len(s), cmaps_f)))

    # Sanity check
    assert word_pad_len == max(list(map(lambda s: len(s), tmaps)))

    padded_wmaps = []
    padded_cmaps_f = []
    padded_cmaps_b = []
    padded_cmarkers_f = []
    padded_cmarkers_b = []
    padded_tmaps = []
    wmap_lengths = []
    cmap_lengths = []

    for w, cf, cb, cmf, cmb, t in zip(wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps):
        # Sanity  checks
        assert len(w) == len(cmf) == len(cmb) == len(t)
        assert len(cmaps_f) == len(cmaps_b)

     
        padded_wmaps.append(w + [word_map['<pad>']] * (word_pad_len - len(w)))
        padded_cmaps_f.append(cf + [char_map['<pad>']] * (char_pad_len - len(cf)))
        padded_cmaps_b.append(cb + [char_map['<pad>']] * (char_pad_len - len(cb)))

        # 0 is always a valid index to pad markers with (-1 is too but torch.gather has some issues with it)
        padded_cmarkers_f.append(cmf + [0] * (word_pad_len - len(w)))
        padded_cmarkers_b.append(cmb + [0] * (word_pad_len - len(w)))

        padded_tmaps.append(t + [tag_map['<pad>']] * (word_pad_len - len(t)))

        wmap_lengths.append(len(w))
        cmap_lengths.append(len(cf))

        # Sanity check
        assert len(padded_wmaps[-1]) == len(padded_tmaps[-1]) == len(padded_cmarkers_f[-1]) == len(
            padded_cmarkers_b[-1]) == word_pad_len
        assert len(padded_cmaps_f[-1]) == len(padded_cmaps_b[-1]) == char_pad_len

    padded_wmaps = torch.LongTensor(padded_wmaps)
    padded_cmaps_f = torch.LongTensor(padded_cmaps_f)
    padded_cmaps_b = torch.LongTensor(padded_cmaps_b)
    padded_cmarkers_f = torch.LongTensor(padded_cmarkers_f)
    padded_cmarkers_b = torch.LongTensor(padded_cmarkers_b)
    padded_tmaps = torch.LongTensor(padded_tmaps)
    wmap_lengths = torch.LongTensor(wmap_lengths)
    cmap_lengths = torch.LongTensor(cmap_lengths)

    return padded_wmaps, padded_cmaps_f, padded_cmaps_b, padded_cmarkers_f, padded_cmarkers_b, padded_tmaps, \
           wmap_lengths, cmap_lengths


def init_embedding(input_embedding):
   
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)




def clip_gradient(optimizer, grad_clip):

    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, model, optimizer, val_f1, word_map, char_map, tag_map, lm_vocab_size, is_best):

    state = {'epoch': epoch,
             'f1': val_f1,
             'model': model,
             'optimizer': optimizer,
             'word_map': word_map,
             'tag_map': tag_map,
             'char_map': char_map,
             'lm_vocab_size': lm_vocab_size}
    filename = 'ner-vi.pth.tar'
    torch.save(state, filename)
    # If checkpoint is the best so far, create a copy to avoid being overwritten by a subsequent worse checkpoint
    if is_best:
        torch.save(state, '/content/drive/My Drive/nlp/BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, new_lr):

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def log_sum_exp(tensor, dim):
    m, _ = torch.max(tensor, dim)
    m_expanded = m.unsqueeze(dim).expand_as(tensor)
    return m + torch.log(torch.sum(torch.exp(tensor - m_expanded), dim))


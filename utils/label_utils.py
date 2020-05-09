#!/usr/bin/python
# encoding: utf-8

import torch


class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'blank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[blank]'] + dict_character  # dummy '[blank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=None):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        # text = ''.join(text)
        # text = [self.dict[char] for char in text]
        d = []
        for s in text:
            t = [self.dict[char] for char in s]
            t.extend([0] * (batch_max_length - len(s)))
            d.append(t)
        return (torch.tensor(d, dtype=torch.long), torch.tensor(length, dtype=torch.long))

    def decode(self, preds, raw=False):
        """ convert text-index into text-label. """
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        result_list = []
        for word, prob in zip(preds_idx, preds_prob):
            if raw:
                result_list.append((''.join([self.character[int(i)] for i in word]), prob))
            else:
                result = []
                conf = []
                for i, index in enumerate(word):
                    if word[i] != 0 and (not (i > 0 and word[i - 1] == word[i])):
                        result.append(self.character[int(index)])
                        conf.append(prob[i])
                result_list.append((''.join(result), conf))
        return result_list


class AttnLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        list_token = ['[GO]', '[s]']  # ['[s]','[UNK]','[PAD]','[GO]']
        list_character = list(character)
        self.character = list_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

    def encode(self, text, batch_max_length):
        """ convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default

        output:
            text : the input of attention decoder. [batch_size x (max_length+2)] +1 for [GO] token and +1 for [s] token.
                text[:, 0] is [GO] token and text is padded with [GO] token after [s] token.
            length : the length of output of attention decoder, which count [s] token also. [3, 7, ....] [batch_size]
        """
        length = [len(s) + 1 for s in text]  # +1 for [s] at end of sentence.
        # batch_max_length = max(length) # this is not allowed for multi-gpu setting
        batch_max_length += 1
        # additional +1 for [GO] at first step. batch_text is padded with [GO] token after [s] token.
        batch_text = torch.zeros(len(text), batch_max_length + 1, dtype=torch.long)
        for i, t in enumerate(text):
            text = list(t)
            text.append('[s]')
            text = [self.dict[char] for char in text]
            batch_text[i][1:1 + len(text)] = torch.Tensor(text).long()  # batch_text[:, 0] = [GO] token
        return (batch_text, torch.Tensor(length).int())

    def decode(self, preds):
        """ convert text-index into text-label. """
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        result_list = []
        for word, prob in zip(preds_idx, preds_prob):
            text = ''.join([self.character[i] for i in word])
            end_idx = text.find('[s]')
            text = text[:end_idx]
            conf = prob[:end_idx].tolist()
            result_list.append((text, conf))
        return result_list

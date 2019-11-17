from seq2seq_loader import *

from random import randint, shuffle
from random import random as rand
import numpy as np
from tqdm import *
import torch
import torch.utils.data
import pickle

from nltk import word_tokenize, pos_tag
from nltk.stem import PorterStemmer


pos_tag_list = ["CC", "CD", "DT", "EX", "FW",
                "IN", "JJ", "JJR", "JJS", "LS",
                "MD", "NN", "NNS", "NNP", "NNPS",
                "PDT", "POS", "PRP", "PRP$", "RB",
                "RBR", "RBS", "RP", "TO", "UH",
                "VB", "VBD", "VBG", "VBN", "VBP",
                "VBZ", "WDT", "WP", "WP$", "WRB"]

pos_tag_dict = {tag:i for i, tag in enumerate(pos_tag_dict)}

pos_tag_lens = 35

def truncate_tokens_pair(tokens_a, tokens_b, pos_seq, pre_seq, max_len, max_len_a=0, max_len_b=0, trunc_seg=None, always_truncate_tail=False):
    num_truncated_a = [0, 0]
    num_truncated_b = [0, 0]
    trunc_flag = "a"
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
            trunc_flag = "a"

        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = tokens_b
            num_truncated = num_truncated_b
            trunc_flag = "b"

        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == 'a':
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
                trunc_flag = "a"
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
                trunc_flag = "b"
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
                trunc_flag = "a"
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
                trunc_flag = "b"

        # whether always truncate source sequences
        if (not always_truncate_tail) and (rand() < 0.5):
            del trunc_tokens[0]
            if trunc_flag == "a":
                del pos_seq[0]
                del pre_seq[0]
            num_truncated[0] += 1
        else:
            trunc_tokens.pop()
            if trunc_flag == "a":
                pos_seq.pop()
                pre_seq.pop()

            num_truncated[1] += 1
    return num_truncated_a, num_truncated_b


class Kp20kDataset(Seq2SeqDataset):
    """ Load sentence pair (sequential or random order) from corpus """

    def __init__(self, file_src, batch_size, tokenizer, max_len, file_oracle=None, short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[]):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.max_len = max_len  # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order
        self.stemmer = PorterStemmer()

        # read the file into memory
        self.ex_list = []
        self.cached = False
        if os.path.exists("cached.pl"):
            self.cached = True
        
        if self.cached:
            print("Loading Documents From cached")
            with open("cached.pl", "rb") as f:
                self.ex_list = pickle.load(f)
        
        else:
            print("Loading Documents From {}".format(file_src))
            with open(file_src, "r", encoding="utf-8") as f_src:
                for line in tqdm(f_src.realines()):
                    line = json.loads(line)
                    doc = line["title"] + " " + line["abstract"]
                    keywords = line["keyword"].replace(";", " ")

                    doc_tk_orig = word_tokenize(doc)

                    # label orig sents with pos_tag and present keyphrase
                    pos_tag_seq_orig = [x[1] for x in pos_tag(doc_tk)]
                    pos_tag_idx_orig = []
                    for x in pos_tag_seq_orig:
                        if x in pos_tag_dict:
                            pos_tag_idx_orig.append(pos_tag_dict[x])
                        else:
                            pos_tag_idx_orig.append(0)
                        
                        keywords_stemmed = " ".join([self.stemmer.stem(x) for x in word_tokenize(keywords)])

                        present_label_idx_orig = [0]*len(doc_tk_orig)

                        for i, tk in enumerate(doc_tk_orig):
                            tk_stemmed = self.stemmer.stem(tk)
                            if tk_stemmed in keywords_stemmed:
                                present_label_idx_orig[i] = 1
                        
                        # split orig_idx to wordpiece idx
                        orig_to_split_map = {}
                        split_cnt = 0
                        doc_tk_split = []
                        pos_tag_idx_split = []
                        present_label_idx_split = []
                        for i, tk in enumerate(doc_tk_orig):
                            tk_pieces = self.tokenizer.tokenize(tk)
                            orig_to_split_map[i] = []
                            for tkk in tk_pieces:
                                doc_tk_split.append(tkk)
                                pos_tag_idx_split.append(pos_tag_idx_orig[i])
                                present_label_idx_orig.append(present_label_idx_orig[i])
                                orig_to_split_map[i].append(split_cnt)
                                split_cnt += 1
                        
                        # find the absent keyphrases 
                        keywords = line["keyword"].split(";")
                        
                        doc_stemmed = " ".join([stemmer.stem(x) for x in word_tokenize(doc)])
                        absent_keyphrase = []

                        for kk in keywords:
                            kk_stemmed = " ".join([stemmer.stem(x) for x in word_tokenize(kk)])
                            if kk_stemmed not in doc_stemmed:
                                absent_keyphrase.append(kk)
                        

                        for kk in absent_keyphrase:
                            tgt_tk = tokenizer.tokenize(tk)
                            if len(tgt_tk) > 0 and len(doc_tk_split) > 0:
                                try:
                                    assert len(doc_tk_split) == len(pos_tag_idx_split) == len(present_label_idx_split)
                                except Exception as e:
                                    print("src length don't match")
                                    print("doc_tk:{}  pos_len:{}  pre_len:{}".format(len(doc_tk_split), len(pos_tag_idx_split), len(present_label_idx_split)))
                                
                                self.ex_list.append((doc_tk_split, tgt_tk, pos_tag_idx_split, present_label_idx_split))
                    
                    # save to cache
                    with open("cached.pl", "wb") as f:
                        pickle.dump(self.ex_list, f)
                    
            print("Load {0} instances".format(len(self.ex_list)))

            # caculate the statistics
            src_tk_lens = [len(x[0]) for x in self.ex_list]
            tgt_tk_lens = [len(x[1]) for x in self.ex_list]

            print("Dataset Statistics")
            print("src_tokens max:{}  min:{}  avg:{}".format(max(src_tk_lens), min(src_tk_lens), sum(src_tk_lens) / len(src_tk_lens)))
            print("tgt_tokens max:{}  min:{}  avg:{}".format(max(tgt_tk_lens), min(tgt_tk_lens), sum(tgt_tk_lens) / len(tgt_tk_lens)))


class Preprocess4Kp20k(Pipeline):
    """ Pre-processing steps for pretraining transformer """


    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, skipgram_prb=0, skipgram_size=0, block_mask=False, mask_whole_word=False, new_segment_ids=False, truncate_config={}, mask_source_words=False, mode="s2s", has_oracle=False, num_qkv=0, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word
        self.new_segment_ids = new_segment_ids
        self.always_truncate_tail = truncate_config.get(
            'always_truncate_tail', False)
        self.max_len_a = truncate_config.get('max_len_a', None)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        self.task_idx = 3   # relax projection layer for different tasks
        self.mask_source_words = mask_source_words
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.has_oracle = has_oracle
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift
    
    def __call__(self, instance):

        tokens_a, tokens_b, pos_seq, pre_seq = instance

        if self.pos_shift:
            tokens_b = ['[S2S_SOS]'] + tokens_b

        # -3  for special tokens [CLS], [SEP], [SEP]
        num_truncated_a, _ = truncate_tokens_pair(tokens_a, tokens_b, pos_seq, pre_seq, self.max_len - 3, max_len_a=self.max_len_a,
                                                  max_len_b=self.max_len_b, trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)

        # Add Special Tokens
        if self.s2s_special_token:
            tokens = ['[S2S_CLS]'] + tokens_a + \
                ['[S2S_SEP]'] + tokens_b + ['[SEP]']
        else:
            tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        
        pos_seq = [0] + pos_seq + [0] 
        pre_seq = [0] + pre_seq + [0] 

        if self.new_segment_ids:
            if self.mode == "s2s":
                if self.s2s_add_segment:
                    if self.s2s_share_segment:
                        segment_ids = [0] + [1] * \
                            (len(tokens_a)+1) + [5]*(len(tokens_b)+1)
                    else:
                        segment_ids = [4] + [6] * \
                            (len(tokens_a)+1) + [5]*(len(tokens_b)+1)
                else:
                    segment_ids = [4] * (len(tokens_a)+2) + \
                        [5]*(len(tokens_b)+1)
            else:
                segment_ids = [2] * (len(tokens))
        else:
            segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)

        if self.pos_shift:
            n_pred = min(self.max_pred, len(tokens_b))
            masked_pos = [len(tokens_a)+2+i for i in range(len(tokens_b))]
            masked_weights = [1]*n_pred
            masked_ids = self.indexer(tokens_b[1:]+['[SEP]'])
        else:
            # For masked Language Models
            # the number of prediction is sometimes less than max_pred when sequence is short
            effective_length = len(tokens_b)
            if self.mask_source_words:
                effective_length += len(tokens_a)
            n_pred = min(self.max_pred, max(
                1, int(round(effective_length*self.mask_prob))))
            # candidate positions of masked tokens
            cand_pos = []
            special_pos = set()
            for i, tk in enumerate(tokens):
                # only mask tokens_b (target sequence)
                # we will mask [SEP] as an ending symbol
                if (i >= len(tokens_a)+2) and (tk != '[CLS]'):
                    cand_pos.append(i)
                elif self.mask_source_words and (i < len(tokens_a)+2) and (tk != '[CLS]') and (not tk.startswith('[SEP')):
                    cand_pos.append(i)
                else:
                    special_pos.add(i)
            shuffle(cand_pos)

            masked_pos = set()
            max_cand_pos = max(cand_pos)
            for pos in cand_pos:
                if len(masked_pos) >= n_pred:
                    break
                if pos in masked_pos:
                    continue

                def _expand_whole_word(st, end):
                    new_st, new_end = st, end
                    while (new_st >= 0) and tokens[new_st].startswith('##'):
                        new_st -= 1
                    while (new_end < len(tokens)) and tokens[new_end].startswith('##'):
                        new_end += 1
                    return new_st, new_end

                if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                    # ngram
                    cur_skipgram_size = randint(2, self.skipgram_size)
                    if self.mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(
                            pos, pos + cur_skipgram_size)
                    else:
                        st_pos, end_pos = pos, pos + cur_skipgram_size
                else:
                    # directly mask
                    if self.mask_whole_word:
                        st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                    else:
                        st_pos, end_pos = pos, pos + 1

                for mp in range(st_pos, end_pos):
                    if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                        masked_pos.add(mp)
                    else:
                        break

            masked_pos = list(masked_pos)
            if len(masked_pos) > n_pred:
                shuffle(masked_pos)
                masked_pos = masked_pos[:n_pred]

            masked_tokens = [tokens[pos] for pos in masked_pos]
            for pos in masked_pos:
                if rand() < 0.8:  # 80%
                    tokens[pos] = '[MASK]'
                elif rand() < 0.5:  # 10%
                    tokens[pos] = get_random_word(self.vocab_words)
            # when n_pred < max_pred, we only calculate loss within n_pred
            masked_weights = [1]*len(masked_tokens)

            # Token Indexing
            masked_ids = self.indexer(masked_tokens)
        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)


        if self.num_qkv > 1:
            mask_qkv = [0]*(len(tokens_a)+2) + [1] * (len(tokens_b)+1)
            mask_qkv.extend([0]*n_pad)
        else:
            mask_qkv = None

        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        if self.mode == "s2s":
            input_mask[:, :len(tokens_a)+2].fill_(1)
            second_st, second_end = len(
                tokens_a)+2, len(tokens_a)+len(tokens_b)+3
            input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])
        else:
            st, end = 0, len(tokens_a) + len(tokens_b) + 3
            input_mask[st:end, st:end].copy_(self._tril_matrix[:end, :end])

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            if masked_ids is not None:
                masked_ids.extend([0]*n_pad)
            if masked_pos is not None:
                masked_pos.extend([0]*n_pad)
            if masked_weights is not None:
                masked_weights.extend([0]*n_pad)

        oracle_pos = None
        oracle_weights = None
        oracle_labels = None
        if self.has_oracle:
            s_st, labls = instance[2:]
            oracle_pos = []
            oracle_labels = []
            for st, lb in zip(s_st, labls):
                st = st - num_truncated_a[0]
                if st > 0 and st < len(tokens_a):
                    oracle_pos.append(st)
                    oracle_labels.append(lb)
            oracle_pos = oracle_pos[:20]
            oracle_labels = oracle_labels[:20]
            oracle_weights = [1] * len(oracle_pos)
            if len(oracle_pos) < 20:
                x_pad = 20 - len(oracle_pos)
                oracle_pos.extend([0] * x_pad)
                oracle_labels.extend([0] * x_pad)
                oracle_weights.extend([0] * x_pad)

            return (input_ids, segment_ids, input_mask, mask_qkv, masked_ids,
                    masked_pos, masked_weights, -1, self.task_idx,
                    oracle_pos, oracle_weights, oracle_labels)
        
        try:
            pass
        except Exception as e:
            print("input_ids length doesn not match pos_seq length and pre seq length")
            print("input_ids: {}, pos_seq: {}, pre_seq:{}".format(len(input_ids), len(pos_seq), len(pre_seq)))

        return (input_ids, segment_ids, pos_seq, pre_seq, input_mask, mask_qkv, masked_ids, masked_pos, masked_weights, -1, self.task_idx)


                
                
                
    
               


               
            
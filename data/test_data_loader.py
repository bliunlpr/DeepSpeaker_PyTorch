import os
import random
import gzip
import numpy as np

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data.sampler import Sampler
import math

from data.base_data_loader import BaseDataset


def create_pair_indices(pair_txt):   
    pairID = []
    inds = dict()
    pair_num = 0
    with open(pair_txt) as f:
        pairs = f.readlines()
        for pair in pairs:
            pair_splits = pair.split(' ')
            if len(pair_splits) < 5:
                continue
            pairID.append(pair_num)
            label = int(pair_splits[-1])
            test_feat_path = pair_splits[-2].strip()
            test_utt_id = pair_splits[-3].strip()
            
            enroll_num = int((len(pair_splits) - 3) / 2)
            enroll_utt_id_list = []
            enroll_feat_path_list = []
            for x in range(enroll_num):
                enroll_utt_id_list.append(pair_splits[2*x].strip())
                enroll_feat_path_list.append(pair_splits[2*x+1].strip()) 
            inds[pair_num] = (enroll_utt_id_list, enroll_feat_path_list, test_utt_id, test_feat_path, label)
            #inds[pair_num] = (pair_splits[0].strip(), pair_splits[1].strip(), pair_splits[2].strip(), pair_splits[3].strip(), int(pair_splits[4]))
            pair_num += 1
    return pairID, inds, len(pairID)


class DeepSpeakerTestDataset(BaseDataset):
    def __init__(self, opt, data_dir):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        utt_id /path/to/audio.wav
        ...
        :param data_scp: Path to scp as describe above
        :param label_file : Dictionary containing the delta_order, context_width, normalize_type and max_num_utt_cmvn
        :param audio_conf: Dictionary containing the sample_rate, num_channel, window_size window_shift
        """                            
        self.pairID, self.pair_indices, self.pair_utt_size = create_pair_indices(os.path.join(data_dir, 'pairs.txt'))
        for i in range(len(self.pair_indices)):
            enroll_utt_id_list, enroll_feat_path_list, test_utt_id, test_feat_path, label = self.pair_indices[i]
            in_feat = self.load_feat(test_feat_path, opt.delta_order)
            if in_feat is not None:
                break
        self.feat_size = in_feat.shape[1]
        self.in_size = self.feat_size * (opt.left_context_width + opt.right_context_width + 1)
        self.num_speaker = len(self.pair_indices)
        print('have {} pair of speakers'.format(len(self.pair_indices)))
        
        super(DeepSpeakerTestDataset, self).__init__(opt, data_dir)
            
    def __getitem__(self, index):
        pair = self.pairID[index]
        enroll_utt_id_list, enroll_feat_path_list, test_utt_id, test_feat_path, label = self.pair_indices[pair]
        enroll_num = len(enroll_utt_id_list)
        enroll_feature_mat_list = []
        for x in range(enroll_num):
            enroll_utt_id = enroll_utt_id_list[x]
            enroll_feat_path = enroll_feat_path_list[x]
            if enroll_utt_id in self.utt2vad.keys():
                vad_idx = self.utt2vad[enroll_utt_id]
            else:
                vad_idx = None
            feature_mat = self.parse_feat(enroll_feat_path, self.delta_order, vad_idx, self.cmvn, self.left_context_width,
                                          self.right_context_width)
            feature_mat = torch.FloatTensor(feature_mat)
            enroll_feature_mat_list.append(feature_mat)

        if test_utt_id in self.utt2vad.keys():
            vad_idx = self.utt2vad[test_utt_id]
        else:
            vad_idx = None
        test_feature_mat = self.parse_feat(test_feat_path, self.delta_order, vad_idx, self.cmvn, self.left_context_width,
                                           self.right_context_width) 
        utt_id_list = enroll_utt_id_list + [test_utt_id]
        return utt_id_list, enroll_feature_mat_list, torch.FloatTensor(test_feature_mat), torch.IntTensor([label])
        
    def __len__(self):
        return self.num_speaker

class DeepSpeakerTestDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(DeepSpeakerTestDataLoader, self).__init__(*args, **kwargs)
        
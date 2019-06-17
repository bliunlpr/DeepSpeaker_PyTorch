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


class DeepSpeakerSeqDataset(BaseDataset):
    def __init__(self, opt, data_dir):
        feats_scp_file = os.path.join(data_dir, 'feats.scp')
        scp_dict = {}
        self.feats_scp = []
        with open(feats_scp_file, 'r') as scp_reader:
            for line in scp_reader:
                line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
                splits = line.split()
                if len(splits) < 2:
                    continue
                utt_id = splits[0]
                utt_path = splits[1]
                scp_dict[utt_id] = utt_path
                self.feats_scp.append((utt_id, utt_path))

        self.num_utt = len(scp_dict)
        for i in range(self.num_utt):
            in_feat = self.load_feat(self.feats_scp[i][1], opt.delta_order)
            if in_feat is not None:
                break
        self.feat_size = in_feat.shape[1]
        self.in_size = self.feat_size * (opt.left_context_width + opt.right_context_width + 1)

        utt2data_file = os.path.join(data_dir, 'utt2data')
        utt2data_dict = {}
        with open(utt2data_file, 'r') as utt2data_reader:
            for line in utt2data_reader:
                line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
                splits = line.split()
                if len(splits) < 2:
                    continue
                utt_id = splits[0]
                data_id = splits[1]
                utt2data_dict[utt_id] = data_id

        utt2spk_file = os.path.join(data_dir, 'utt2spk')
        utt2spk_dict = {}
        with open(utt2spk_file, 'r') as utt2spk_reader:
            for line in utt2spk_reader:
                line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
                splits = line.split()
                if len(splits) < 2:
                    continue
                utt_id = splits[0]
                spk_id = splits[1]
                utt2spk_dict[utt_id] = spk_id

        spk2utt_file = os.path.join(data_dir, 'spk2utt')
        spk2utt = {}
        spk2idx = {}
        num_speaker = 0
        with open(spk2utt_file) as spk2utt_reader:
            for line in spk2utt_reader.readlines():
                line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
                splits = line.split()
                if len(splits) < 2:
                    continue
                spk_id = splits[0]
                if spk_id not in spk2idx.keys():
                    spk2idx[spk_id] = num_speaker
                    num_speaker = num_speaker + 1
                if spk_id not in spk2utt.keys():
                    spk2utt[spk_id] = []
                for utt_id in splits[1:]:
                    if utt_id in scp_dict.keys() and utt_id in utt2data_dict.keys():
                        utt_path = scp_dict[utt_id]
                        data_id = utt2data_dict[utt_id]
                        spk2utt[spk_id].append((utt_id, utt_path, data_id))

        data2spk = {}
        data2idx = {}
        num_data = 0
        data2spk_file = os.path.join(data_dir, 'data2spk')
        with open(data2spk_file) as data2spk_reader:
            for line in data2spk_reader.readlines():
                line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
                splits = line.split()
                if len(splits) < 2:
                    continue
                data_id = splits[0]
                if data_id not in data2idx.keys():
                    data2idx[data_id] = num_data
                    num_data = num_data + 1
                if data_id not in data2spk:
                    data2spk[data_id] = []
                for spk_id in splits[1:]:
                    data2spk[data_id].append(spk_id)

        self.spk2ids = spk2idx
        self.spk2utt = spk2utt
        self.data2spk = data2spk
        self.data2idx = data2idx
        self.num_speaker = num_speaker
        self.class_nums = self.num_speaker
        self.num_data = num_data
        num_speaker = 0
        for data_id, data2spk in self.data2spk.items():
            num_speaker = num_speaker + len(data2spk)
        self.data_sampler = {}
        speaker_data = 0
        for data_id, data2spk in self.data2spk.items():
            speaker_data = speaker_data + len(data2spk)
            self.data_sampler[data_id] = speaker_data / num_speaker
        print('The training dataset have {} dataset and {} speakers'.format(self.num_data, self.num_speaker))
            
        super(DeepSpeakerSeqDataset, self).__init__(opt, data_dir)

    def __getitem__(self, index):
        
        segment_win = np.random.randint(self.min_segment_length, self.max_segment_length)
        segment_shift = int(np.round(segment_win * self.segment_shift_rate))
        num_segment_utt = np.random.randint(self.min_num_segment, self.max_num_segment)
                    
        utter_batch = []
        utter_spk_ids_batch = []
        utter_segment_num_batch = []

        rand_ratio = np.random.random_sample()
        i = 0
        for data_id, ratio in self.data_sampler.items():
            if i == 0:
                selected_data_id = data_id
            i = i + 1
            if rand_ratio <= ratio:
                break
            selected_data_id = data_id
        
        speakers = self.data2spk[selected_data_id]
        num_speaker = len(speakers)
        rand_idx = np.random.permutation(num_speaker)
        
        ispeaker = 0
        while ispeaker < self.num_speaker_batch:
            ispeaker = ispeaker % num_speaker
            selected_speaker = speakers[rand_idx[ispeaker]]
            selected_file = self.spk2utt[selected_speaker]
            num_utts_spk = len(selected_file)
            idx_utts_spk = np.random.permutation(num_utts_spk)

            selected_utts_feature = []
            selected_utts_num = 0
            for i in idx_utts_spk:
                utt_id, feat_path, data_id = selected_file[i]
                if utt_id in self.utt2vad.keys():
                    vad_idx = self.utt2vad[utt_id]
                else:
                    vad_idx = None
                
                feature_mat = self.parse_feat(feat_path, self.delta_order, vad_idx, self.cmvn,
                                               self.left_context_width, self.right_context_width)
                if feature_mat is None or feature_mat.shape[0] - segment_win <= segment_shift:
                    continue

                selected_utts_num = selected_utts_num + 1
                selected_utts_feature.append(feature_mat)
                if selected_utts_num > self.num_utts_speaker:
                    break
            if selected_utts_num <= 0:
                continue
            feature_mats = np.zeros(shape=[0, segment_win, self.in_size], dtype=np.float32)
            spk_id_mats = np.zeros(shape=[0], dtype=np.int64)
            for num in range(self.num_utts_speaker):
                num = num % selected_utts_num
                feature_mat = selected_utts_feature[num]
                
                num_frame = feature_mat.shape[0]
                start_frame = 0
                end_frame = num_frame
                
                num_frame_needed = (num_segment_utt - 1) * (segment_win - segment_shift) + segment_win
                if num_frame_needed + 1 < num_frame:
                    start_frame = np.random.randint(0, num_frame - num_frame_needed - 1)
                    end_frame = start_frame + num_frame_needed
                
                segment_num = 0
                for start in range(start_frame, end_frame, segment_shift):
                    end = start + segment_win
                    if end < feature_mat.shape[0] and segment_num < num_segment_utt:
                        feature_mat_slice = feature_mat[start:end, :]
                        feature_mat_slice = feature_mat_slice[np.newaxis, :]
                        feature_mats = np.concatenate((feature_mats, feature_mat_slice), axis=0)
                        spk_id_mat = np.array([self.spk2ids[selected_speaker]], dtype=np.int64)
                        ##spk_id_mat = spk_id_mat[np.newaxis, :]
                        spk_id_mats = np.concatenate((spk_id_mats, spk_id_mat), axis=0)
                        segment_num += 1
                utter_segment_num_batch.append(segment_num)
            utter_batch.append(feature_mats)
            utter_spk_ids_batch.append(spk_id_mats)
            ispeaker = ispeaker + 1
            del feature_mats, spk_id_mats
        utter_batch = np.concatenate(utter_batch, axis=0)         # utterance batch [batch(NM), frames, n_mels]
        utter_batch = np.transpose(utter_batch, axes=(1,0,2))     # transpose [frames, batch, n_mels]  
        utter_segment_num_batch = np.array(utter_segment_num_batch, dtype=np.int64)
        utter_spk_ids_batch = np.concatenate(utter_spk_ids_batch, axis=0)
        return torch.FloatTensor(utter_batch), torch.LongTensor(utter_segment_num_batch), torch.LongTensor(utter_spk_ids_batch)

    def __len__(self):
        return self.num_speaker

class DeepSpeakerSeqDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(DeepSpeakerSeqDataLoader, self).__init__(*args, **kwargs)

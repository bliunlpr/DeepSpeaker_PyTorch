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

from data import kaldi_io


def splice(utt, left_context_width, right_context_width):
    """
    splice the utterance
    Args:
        utt: numpy matrix containing the utterance features to be spliced
        context_width: how many frames to the left and right should
            be concatenated
    Returns:
        a numpy array containing the spliced features, if the features are
        too short to splice None will be returned
    """
    # return None if utterance is too short
    if utt.shape[0] < 1 + left_context_width + right_context_width:
        return None

    #  create spliced utterance holder
    utt_spliced = np.zeros(
        shape=[utt.shape[0], utt.shape[1] * (1 + left_context_width + right_context_width)],
        dtype=np.float32)

    #  middle part is just the utterance
    utt_spliced[:, left_context_width * utt.shape[1]:
                   (left_context_width + 1) * utt.shape[1]] = utt

    for i in range(left_context_width):
        #  add left context
        utt_spliced[i + 1:utt_spliced.shape[0],
        (left_context_width - i - 1) * utt.shape[1]:
        (left_context_width - i) * utt.shape[1]] = utt[0:utt.shape[0] - i - 1, :]

    for i in range(right_context_width):
        # add right context
        utt_spliced[0:utt_spliced.shape[0] - i - 1,
        (left_context_width + i + 1) * utt.shape[1]:
        (left_context_width + i + 2) * utt.shape[1]] = utt[i + 1:utt.shape[0], :]

    return utt_spliced


def add_delta(utt, delta_order):
    num_frames = utt.shape[0]
    feat_dim = utt.shape[1]

    utt_delta = np.zeros(
        shape=[num_frames, feat_dim * (1 + delta_order)],
        dtype=np.float32)

    #  first order part is just the utterance max_offset+1
    utt_delta[:, 0:feat_dim] = utt

    scales = [[1.0], [-0.2, -0.1, 0.0, 0.1, 0.2],
              [0.04, 0.04, 0.01, -0.04, -0.1, -0.04, 0.01, 0.04, 0.04]]

    delta_tmp = np.zeros(shape=[num_frames, feat_dim], dtype=np.float32)
    for i in range(1, delta_order + 1):
        max_offset = (len(scales[i]) - 1) / 2
        for j in range(-max_offset, 0):
            delta_tmp[-j:, :] = utt[0:(num_frames + j), :]
            for k in range(-j):
                delta_tmp[k, :] = utt[0, :]
            scale = scales[i][j + max_offset]
            if scale != 0.0:
                utt_delta[:, i * feat_dim:(i + 1) * feat_dim] += scale * delta_tmp

        scale = scales[i][max_offset]
        if scale != 0.0:
            utt_delta[:, i * feat_dim:(i + 1) * feat_dim] += scale * utt

        for j in range(1, max_offset + 1):
            delta_tmp[0:(num_frames - j), :] = utt[j:, :]
            for k in range(j):
                delta_tmp[-(k + 1), :] = utt[(num_frames - 1), :]
            scale = scales[i][j + max_offset]
            if scale != 0.0:
                utt_delta[:, i * feat_dim:(i + 1) * feat_dim] += scale * delta_tmp
    return utt_delta


def dump_to_text(spk2ids, out_file):
    fwrite = open(out_file, 'w')
    for key, value in spk2ids.items():
        fwrite.write('{} {}'.format(value, key) + '\n')
    fwrite.close()


def make_utt2spk(utt2spk_file, utt_ids):
    utt2spk = {}
    speakers = []
    fread = open(utt2spk_file, 'r')
    for line in fread.readlines():
        line = line.replace('\n','').strip()
        splits = line.split(' ')
        utt_id = splits[0]
        spk_id = splits[1]
        utt2spk[utt_id] = spk_id
        speakers += [spk_id]

    speakers = list(set(speakers))
    speakers.sort()
    spk2ids = {v: i for i, v in enumerate(speakers)}

    utt2spk_ids = {}
    for sample in utt_ids:
        utt_id, utt_path = sample[0], sample[1]
        try:
            speaker = spk2ids[utt2spk[utt_id]]
            utt2spk_ids[utt_id] = speaker
        except:
            print ('{} has no utt2spk '.format(utt_id))
    return utt2spk_ids, spk2ids
    
    
def create_indices(wav_utt_ids, utt2spk_ids):
    inds = dict()
    for sample in wav_utt_ids:
        utt_id, audio_path = sample[0], sample[1]
        spk_id = utt2spk_ids[utt_id]
        if spk_id not in inds:        
            inds[spk_id] = []
        inds[spk_id].append((utt_id, audio_path))
    return inds


class BaseDataset(Dataset):
    def __init__(self, opt, data_dir):
        self.opt = opt
        self.exp_path = opt.exp_path

        self.num_utt_cmvn = opt.num_utt_cmvn
        self.normalize_type = opt.normalize_type
        self.cmvn = None

        self.num_speaker_batch = opt.speaker_num
        self.num_utts_speaker = opt.utter_num
        
        self.min_num_segment = opt.min_num_segment
        self.max_num_segment = opt.max_num_segment
        self.min_segment_length = opt.min_segment_length
        self.max_segment_length = opt.max_segment_length
        self.segment_shift_rate = opt.segment_shift_rate
        
        self.delta_order         = opt.delta_order
        self.left_context_width  = opt.left_context_width
        self.right_context_width = opt.right_context_width
        
        self.utt2vad = {}
        self.vad = opt.vad
        if self.vad:
            utt2vad_file = os.path.join(data_dir, 'utt2vad')
            with open(utt2vad_file) as utt2vad_reader:
                for line in utt2vad_reader.readlines():
                    line = line.strip().replace('\n', '').replace('\r', '').replace('\t', ' ')
                    splits = line.split()
                    if len(splits) < 2:
                        continue
                    utt_id = splits[0]
                    self.utt2vad[utt_id] = []
                    for vad_idx in splits[1:]:
                        splits = vad_idx.split(':')
                        start = splits[0]
                        end = splits[1]
                        self.utt2vad[utt_id].append((float(start), float(end)))
        
        if self.normalize_type == 1:
            if opt.cmvn_file is None:
                self.cmvn_file = os.path.join(self.exp_path, 'cmvn.npy')
            else:
                self.cmvn_file = opt.cmvn_file
            self.cmvn = self.loading_cmvn(self.cmvn_file)
            
        super(BaseDataset, self).__init__()

    def load_feat(self, feat_path, delta_order=0):
        try:
            if "ark:" in feat_path:
                feat = kaldi_io.read_mat(feat_path)
            else:
                feat = np.load(feat_path)

            if feat is not None and delta_order > 0:
                feat = add_delta(feat, delta_order)
        except:
            print('{} has error'.format(feat_path))
            feat = None
        return feat

    def perform_vad(self, feat, vad_idx):
        num_frame = feat.shape[0]
        num_feat = feat.shape[1]
        vad_feat = np.zeros(shape=[0, num_feat], dtype=np.float32)
        for start_ratio, end_ratio in vad_idx:
            start_idx = int(np.min([np.max([np.round(start_ratio * num_frame), 0]), num_frame]))
            end_idx = int(np.min([np.max([np.round(end_ratio * num_frame), 0]), num_frame]))
            vad_feat = np.concatenate((vad_feat, feat[start_idx:end_idx, :]), axis=0)
        return vad_feat

    def rand_segment(self, feat, segment_length):
        if feat is None:
            return None
        num_frame = feat.shape[0]
        if segment_length < num_frame:
            index = random.randint(0, (num_frame - segment_length) - 1)
            segment_feat = feat[index:index + segment_length, :]
        else:
            segment_feat = None
        return segment_feat

    def parse_feat(self, feat_path, delta_order=0, vad_idx=None, cmvn=None, left_context_width=0, right_context_width=0):
        if feat_path is None:
            return None
        feat = self.load_feat(feat_path, delta_order)
        if feat is None:
            return None

        if vad_idx is not None:
            self.perform_vad(feat, vad_idx)

        if cmvn is not None and self.normalize_type == 1:
            feat = (feat + cmvn[0, :]) * cmvn[1, :]
            
        if left_context_width > 0 or right_context_width > 0:
            feat = splice(feat, left_context_width, right_context_width)

        return feat

    def compute_cmvn(self):
        cmvn_num = min(self.num_utt, self.num_utt_cmvn)
        print(">> compute cmvn using {0} utterance ".format(cmvn_num))

        sum = np.zeros(shape=[1, self.feat_size], dtype=np.float32)
        sum_sq = np.zeros(shape=[1, self.feat_size ], dtype=np.float32)
        cmvn = np.zeros(shape=[2, self.feat_size], dtype=np.float32)

        frame_count = 0
        cmvn_rand_idx = np.random.permutation(self.num_utt)
        for n in tqdm(range(cmvn_num)):
            feat_path = self.feats_scp[cmvn_rand_idx[n]][1]
            feature_mat = self.load_feat(feat_path, self.delta_order)

            if feature_mat is None:
                continue

            sum_1utt = np.sum(feature_mat, axis=0)
            sum = np.add(sum, sum_1utt)
            feature_mat_square = np.square(feature_mat)
            sum_sq_1utt = np.sum(feature_mat_square, axis=0)
            sum_sq = np.add(sum_sq, sum_sq_1utt)
            frame_count += feature_mat.shape[0]
        mean = sum / frame_count
        var = sum_sq / frame_count - np.square(mean)
        cmvn[0, :] = -mean
        cmvn[1, :] = 1.0 / np.sqrt(var)
        return cmvn

    def loading_cmvn(self, cmvn_file):
        if os.path.exists(cmvn_file):
            cmvn = np.load(cmvn_file)
            if cmvn.shape[1] == self.feat_size:
                print ('load cmvn from {}'.format(cmvn_file))
            else:
                cmvn = self.compute_cmvn()
                np.save(cmvn_file, cmvn)
                print ('original cmvn is wrong, so save new cmvn to {}'.format(cmvn_file))
        else:
            cmvn = self.compute_cmvn()
            np.save(cmvn_file, cmvn)
            print ('save cmvn to {}'.format(cmvn_file))
        return cmvn
        

class DeepSpeakerDataset(BaseDataset):
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
        super(DeepSpeakerDataset, self).__init__(opt, data_dir)

    def __getitem__(self, index):
        if self.data_type == 'train': 
            segment_win = np.random.randint(self.min_segment_length, self.max_segment_length)
        else:
            segment_win = None
            
        if self.data_type != 'train':
            pair = self.pairID[index]
            utt_id0, utt_id1, feat_path0, feat_path1, label = self.pair_indices[pair]

            if utt_id0 in self.utt2vad.keys():
                vad_idx0 = self.utt2vad[utt_id0]
            else:
                vad_idx0 = None
            feature_mat0 = self.parse_feat(feat_path0, self.delta_order, vad_idx0, self.cmvn, self.left_context_width,
                                           self.right_context_width)

            if utt_id1 in self.utt2vad.keys():
                vad_idx1 = self.utt2vad[utt_id1]
            else:
                vad_idx1 = None
            feature_mat1 = self.parse_feat(feat_path1, self.delta_order, vad_idx1, self.cmvn, self.left_context_width,
                                           self.right_context_width)
            utt_id0 = utt_id0.replace('__', '_')
            utt_id1 = utt_id1.replace('__', '_')
            return '{}__{}'.format(utt_id0, utt_id1), torch.FloatTensor(feature_mat0), torch.FloatTensor(
                feature_mat1), torch.IntTensor([label])
        else:
            utter_batch = []
            utter_spk_ids_batch = []

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
                    if feature_mat is None or feature_mat.shape[0] - segment_win <= 0:
                        continue

                    selected_utts_num = selected_utts_num + 1
                    selected_utts_feature.append(feature_mat)
                    if selected_utts_num > self.num_utts_speaker:
                        break
                if selected_utts_num <= 0:
                    continue

                feature_mats = np.zeros(shape=[0, segment_win, self.in_size], dtype=np.float32)
                spk_id_mats = np.zeros(shape=[0, segment_win], dtype=np.int64)
                for num in range(self.num_utts_speaker):
                    num = num % selected_utts_num
                    feature_mat = selected_utts_feature[num]
                    feature_mat_slice = self.rand_segment(feature_mat, segment_win)
                    feature_mat_slice = feature_mat_slice[np.newaxis, :]
                    feature_mats = np.concatenate((feature_mats, feature_mat_slice), axis=0)

                    spk_id_mat = np.array([self.spk2ids[selected_speaker]] * segment_win, dtype=np.int64)
                    spk_id_mat = spk_id_mat[np.newaxis, :]
                    spk_id_mats = np.concatenate((spk_id_mats, spk_id_mat), axis=0)

                utter_batch.append(feature_mats)
                utter_spk_ids_batch.append(spk_id_mats)
                ispeaker = ispeaker + 1
                del feature_mats, spk_id_mats
            utter_batch = np.concatenate(utter_batch, axis=0)  # utterance batch [batch(NM), frames, n_mels]
            utter_batch = np.transpose(utter_batch, axes=(1, 0, 2))  # transpose [frames, batch, n_mels]
            utter_spk_ids_batch = np.concatenate(utter_spk_ids_batch, axis=0)
            return torch.FloatTensor(utter_batch), torch.LongTensor(utter_spk_ids_batch)

    def __len__(self):
        return self.num_speaker

class DeepSpeakerDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(DeepSpeakerDataLoader, self).__init__(*args, **kwargs)

class DeepSpeakerSeqDataset(BaseDataset):
    def __init__(self, opt, data_dir):
        super(DeepSpeakerSeqDataset, self).__init__(opt, data_dir)

    def __getitem__(self, index):
        if self.data_type == 'train': 
            segment_win = np.random.randint(self.min_segment_length, self.max_segment_length)
            segment_shift = int(np.round(segment_win * self.segment_shift_rate))
        else:
            segment_win = None
            segment_shift = 0
            
        if self.data_type == 'train': 
            num_segment_utt = np.random.randint(self.min_num_segment, self.max_num_segment)
        else:
            num_segment_utt = None
            
        if self.data_type != 'train':  
            pair = self.pairID[index]
            utt_id0, utt_id1, feat_path0, feat_path1, label = self.pair_indices[pair]

            if utt_id0 in self.utt2vad.keys():
                vad_idx0 = self.utt2vad[utt_id0]
            else:
                vad_idx0 = None
            feature_mat0 = self.parse_feat(feat_path0, self.delta_order, vad_idx0, self.cmvn, self.left_context_width,
                                           self.right_context_width)

            if utt_id1 in self.utt2vad.keys():
                vad_idx1 = self.utt2vad[utt_id1]
            else:
                vad_idx1 = None
            feature_mat1 = self.parse_feat(feat_path1, self.delta_order, vad_idx1, self.cmvn, self.left_context_width,
                                           self.right_context_width)
            utt_id0 = utt_id0.replace('__', '_')
            utt_id1 = utt_id1.replace('__', '_')
            return '{}__{}'.format(utt_id0, utt_id1), torch.FloatTensor(feature_mat0), torch.FloatTensor(
                feature_mat1), torch.IntTensor([label])
        else:
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
                spk_id_mats = np.zeros(shape=[0, segment_win], dtype=np.int64)
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
                            spk_id_mat = np.array([self.spk2ids[selected_speaker]] * segment_win, dtype=np.int64)
                            spk_id_mat = spk_id_mat[np.newaxis, :]
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

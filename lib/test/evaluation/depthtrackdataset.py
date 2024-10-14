import numpy as np
import os
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class DepthTrackDataset(BaseDataset):
    """
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.depthtrack_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)


        ground_truth_rect_f = []
        for i in range(ground_truth_rect.shape[0]):
            x, y, w, h = ground_truth_rect[i]
            if np.isnan(x):
                ground_truth_rect_f.append([-1, -1, -1, -1])
            else:
                ground_truth_rect_f.append([x, y, w, h])
        ground_truth_rect = np.array(ground_truth_rect_f)


        # frames_path = '{}/{}/color'.format(self.base_path, sequence_name)
        # frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        frames_path = '{}/{}/depth'.format(self.base_path, sequence_name)
        frames_list = ['{}/{:08d}.png'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        return Sequence(sequence_name, frames_list, 'depthtrack', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        with open(os.path.join(self.base_path, 'list.txt'), 'r') as f:
            sequence_list = f.read().splitlines()

        return sequence_list

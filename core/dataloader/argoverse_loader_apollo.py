import os
from os.path import join as pjoin
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset


class DatasetApollo(Dataset):

    def __init__(self, root_dir, sample_size=-1):
        data_lens = os.listdir(root_dir)
        if (sample_size > 0 and sample_size < len(data_lens)):
            data_lens = data_lens[:sample_size]
        self.dataset_path = [pjoin(root_dir, x) for x in data_lens]

    def __len__(self):
        return len(self.dataset_path)

    def __getitem__(self, idx):
        raw_data = pd.read_pickle(self.dataset_path[idx])
        agt_traj_obs, agt_traj_obs_steps = self.process_target_obstacle_pos(raw_data)
        
        obstacle_feats, traj_cnt, vector_mask, obs_p_id = self.process_all_obstacle_feats(raw_data)
        map_feats, vector_mask, map_p_id = self.process_all_map_feats(raw_data, traj_cnt, vector_mask)
        
        vector_data = np.vstack([obstacle_feats, map_feats])
        polyline_mask = np.zeros(450)
        polyline_id = np.vstack([obs_p_id, map_p_id])

        if vector_data.shape[0] < 450:
            pad_vector_data = np.zeros((450-vector_data.shape[0], 50, 9))
            pad_polyline_id = np.zeros((450-vector_data.shape[0], 2))

            polyline_mask[vector_data.shape[0]:] = 1
            vector_data = np.vstack([vector_data, pad_vector_data])
            polyline_id = np.vstack([polyline_id, pad_polyline_id])

        tensor_agt_traj_obs = torch.from_numpy(agt_traj_obs).float()
        tensor_agt_traj_obs_steps = torch.from_numpy(agt_traj_obs_steps).float()
        tensor_vector_data = torch.from_numpy(vector_data).float()
        tensor_vector_mask = torch.from_numpy(vector_mask).bool()
        tensor_polyline_mask = torch.from_numpy(polyline_mask).bool()
        tensor_rand_mask = torch.zeros(450)
        tensor_polyline_id = torch.from_numpy(polyline_id).float()

        # target_obstacle_pos = torch.rand(20, 2)
        # target_obstacle_pos_step = torch.rand(20, 2)
        # vector_data = torch.rand(450, 50, 9)
        # vector_mask = torch.rand(450, 50) > 0.9
        # polyline_mask = torch.rand(450) > 0.9
        # rand_mask = torch.zeros(450)
        # polyline_id = torch.rand(450, 2)

        data = (tensor_agt_traj_obs, 
                tensor_agt_traj_obs_steps, 
                tensor_vector_data, tensor_vector_mask, 
                tensor_polyline_mask, tensor_rand_mask, 
                tensor_polyline_id)

        return {
            'input': data, 
            'gt_trajectory': raw_data['gt_preds'].values[0][0],
            'orig': raw_data['orig'][0], 
            'rot': raw_data['rot'][0]}

    def process_target_obstacle_pos(self, raw_data):
        agt_traj_obs = raw_data['feats'].values[0][0][:, :2]
        agt_traj_obs_steps = np.vstack([np.array([[0, 0]]), agt_traj_obs[1:, :] - agt_traj_obs[:-1, :]])
        return agt_traj_obs, agt_traj_obs_steps

    def process_all_obstacle_feats(self, data_seq):
        feats = np.empty((0, 50, 9))
        vector_mask = np.zeros((450,50))
        obs_p_id = np.zeros((0, 2))

        # get traj features
        traj_feats = data_seq['feats'].values[0]
        traj_has_obss = data_seq['has_obss'].values[0]
        traj_cnt = 0

        for i, [feat, has_obs] in enumerate(zip(traj_feats, traj_has_obss)):
            xy_s = feat[has_obs][:-1, :2]
            xy_n = feat[has_obs][1:, :2]
            
            feat_length = np.ones((len(xy_s), 1)) * 5
            feat_width = np.ones((len(xy_s), 1)) * 2
            obs_attr_1  = np.ones((len(xy_s), 1)) * (11 if i == 0 else 10)
            obs_attr_2 = np.ones((len(xy_s), 1)) * 4
            obs_id = np.ones((len(xy_s), 1)) * traj_cnt

            feat_with_pad = np.vstack([np.zeros((50 - len(xy_s), 9)), np.hstack([xy_s, xy_n, feat_length, feat_width, obs_attr_1, obs_attr_2, obs_id])])
            feats = np.vstack([feats, [feat_with_pad]])
            traj_cnt += 1

            if len(xy_s) > 1:
                vector_mask[i, 1-len(xy_s)] = 1
            else:
                vector_mask[i, -1] = 1
            
            p_id= np.array([feat[has_obs][:, i].min() for i in range(2)])
            obs_p_id = np.vstack([obs_p_id, [p_id]])
            
        return feats, traj_cnt, vector_mask, obs_p_id

    def process_all_map_feats(self, data_seq, traj_cnt, vector_mask):
        feats = np.empty((0, 50, 9))
        map_p_id = np.zeros((0, 2))

        # get lane features
        graph = data_seq['graph'].values[0]
        xy_s = graph['ctrs']
        xy_n = graph['ctrs'] + graph['feats'] / 2
        feat_size = np.zeros((len(xy_s), 2))
        map_attr = graph['intersect'].reshape(-1, 1) * 8
        map_boundary = np.zeros((len(xy_s), 1))
        lane_idcs = graph['lane_idcs'].reshape(-1, 1) + traj_cnt

        feats_origin = np.hstack([xy_s, xy_n, feat_size, map_attr, map_boundary, lane_idcs])

        i = 0
        for idc in np.unique(lane_idcs):
            tmp = feats_origin[lane_idcs.reshape(-1)==idc, :]
            feat = np.vstack([tmp, np.zeros((50 - tmp.shape[0], 9))])
            feats = np.vstack([feats, [feat]])

            vector_mask[i+traj_cnt, tmp.shape[0]:] = 1
            i += 1

            p_id= np.array([tmp[:, i].min() for i in range(2)])
            map_p_id = np.vstack([map_p_id, [p_id]])

        return feats, vector_mask, map_p_id


if __name__ == "__main__":
    test_dataset_dir = "dataset_sample/interm_data_small/train_intermediate/raw"
    model_path = "run/apollo_tnt/vectornet_vehicle_cpu_model.pt"

    dataset = DatasetApollo(root_dir=test_dataset_dir)
    model = torch.jit.load(model_path)

    for i in range(len(dataset)):
        data = dataset[i]
        input = (i.unsqueeze(0) for i in data['input'])
        out = model(input)
        print(out.shape)
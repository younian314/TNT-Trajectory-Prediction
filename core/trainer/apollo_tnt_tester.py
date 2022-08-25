import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import  DataLoader

from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate
from argoverse.evaluation.competition_util import generate_forecasting_h5

from core.util.viz_utils import show_pred_and_gt
from core.dataloader.argoverse_loader_apollo import DatasetApollo



class ApolloTNTTester(object):

    def __init__(self,
                 testset,
                 save_folder: str = "test_apollo_result",
                 model_path: str = "run/apollo_tnt/vectornet_vehicle_cpu_model.pt"
                 ):

        self.testset = testset
        self.save_folder = save_folder
        self.model = torch.jit.load(model_path)

    def test(self,
             miss_threshold=2.0,
             compute_metric=False,
             convert_coordinate=False,
             plot=False,
             save_pred=False):
        """
        test the testset,
        :param miss_threshold: float, the threshold for the miss rate, default 2.0m
        :param compute_metric: bool, whether compute the metric
        :param convert_coordinate: bool, True: under original coordinate, False: under the relative coordinate
        :param save_pred: store the prediction or not, store in the Argoverse benchmark format
        """

        forecasted_trajectories, gt_trajectories = {}, {}

        with torch.no_grad():
        # for data in tqdm(self.test_loader):
            cnt = 0
            for data in tqdm(self.testset):
                input = (i.unsqueeze(0) for i in data['input'])
                out = self.model(input)
                forecasted_trajectory = out.numpy()[0]
                gt_trajectory = data['gt_trajectory']

                if convert_coordinate:
                    forecasted_trajectory = self.convert_coord(forecasted_trajectory, data['orig'], data['rot'])
                    gt_trajectory = self.convert_coord(gt_trajectory, data['orig'], data['rot'])

                forecasted_trajectories[cnt] = [forecasted_trajectory]
                gt_trajectories[cnt] = gt_trajectory
                cnt += 1
        
        # compute the metric
        if compute_metric:
            metric_results = get_displacement_errors_and_miss_rate(
                forecasted_trajectories,
                gt_trajectories,
                1,
                30,
                miss_threshold
            )
            print("[TNTTrainer]: The test result: {};".format(metric_results))


        # plot the result
        if plot:
            fig, ax = plt.subplots()
            for i in range(cnt):
                # ax.set_xlim(-15, 15)
                show_pred_and_gt(ax, gt_trajectories[i], forecasted_trajectories[i])
                plt.pause(3)
                ax.clear()

    # function to convert the coordinates of trajectories from relative to world
    def convert_coord(self, traj, orig, rot):
        traj_converted = np.matmul(np.linalg.inv(rot), traj.T).T + orig.reshape(-1, 2)
        return traj_converted


if __name__ == "__main__":
    # init trainer

    test_dataset_dir = "dataset/interm_data/train_intermediate/raw"
    model_path = "run/apollo_tnt/vectornet_vehicle_cpu_model.pt"
    save_folder = "test_apollo_result"

    test_dataset = DatasetApollo(root_dir=test_dataset_dir, sample_size=100)
    tester = ApolloTNTTester(
        testset=test_dataset,
        model_path=model_path, 
        save_folder=save_folder,
    )
    tester.test(convert_coordinate=True, compute_metric=True, plot=False)


    
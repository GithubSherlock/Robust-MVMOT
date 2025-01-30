import lightning as pl
import os
from torch.utils.data import DataLoader
from typing import Optional
from datasets.multiviewx_dataset import MultiviewX
from datasets.sampler import RandomPairSampler
from datasets.wildtrack_dataset import Wildtrack
from datasets.pedestrian_dataset import PedestrianDataset
from datasets.pedestrian_dataset_cam_dropout import PedestrianDatasetCamDropout


class PedestrianDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "../data/MultiviewX",
            batch_size: int = 1,
            num_workers: int = 4,
            train_cameras=(),
            test_cameras=(),
            resolution=None,
            bounds=None,
            load_depth=False,
            kwargs=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.bounds = bounds
        self.load_depth = load_depth
        self.dataset = os.path.basename(self.data_dir)

        self.train_cameras = train_cameras
        self.test_cameras = test_cameras

        self.data_predict = None
        self.data_test = None
        self.data_val = None
        self.data_train = None
        self.kwargs = kwargs

    def setup(self, stage: Optional[str] = None):
        if 'wildtrack' in self.dataset.lower():
            base = Wildtrack(self.data_dir, train_cameras=self.train_cameras, test_cameras=self.test_cameras)
        elif 'multiviewx' in self.dataset.lower():
            base = MultiviewX(self.data_dir)
        else:
            raise ValueError(f'Unknown dataset name {self.dataset}')

        ## if there ae no kwargs dont do dropout
        if stage == 'fit':
            if self.kwargs is None:
                self.data_train = PedestrianDataset(
                    base,
                    is_train=True,
                    resolution=self.resolution,
                    bounds=self.bounds,
                )
            else:
                self.data_train = PedestrianDatasetCamDropout(
                    base,
                    is_train=True,
                    resolution=self.resolution,
                    bounds=self.bounds,
                    # num_cameras=self.kwargs['num_cameras'],
                    **self.kwargs,
                )
        if stage == 'fit' or stage == 'validate':
            if self.kwargs is None:
                self.data_val = PedestrianDataset(
                    base,
                    is_train=False,
                    resolution=self.resolution,
                    bounds=self.bounds,
                )
            else:
                self.data_val = PedestrianDatasetCamDropout(
                    base,
                    is_train=False,
                    resolution=self.resolution,
                    bounds=self.bounds,
                    # num_cameras=self.kwargs['num_cameras'],
                    **self.kwargs,
                )
        if stage == 'test':
            print(f'\n\ndoing testing with views:{self.test_cameras}\n\n')
            if 'wildtrack' in self.dataset.lower():
                base = Wildtrack(self.data_dir, train_cameras=self.train_cameras, test_cameras=self.test_cameras,
                                 is_test=True)
            if self.kwargs is None:
                self.data_test = PedestrianDataset(
                    base,
                    is_train=False,
                    is_testing=True,
                    resolution=self.resolution,
                    bounds=self.bounds
                )
            else:
                self.data_test = PedestrianDatasetCamDropout(
                    base,
                    is_train=False,
                    is_testing=True,
                    resolution=self.resolution,
                    bounds=self.bounds,
                    # num_cameras=self.kwargs['num_cameras'],
                    **self.kwargs,
                )
        if stage == 'predict':
            self.data_predict = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=RandomPairSampler(self.data_train)
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

from pytorch_lightning.callbacks import TQDMProgressBar
import sys

class CustomProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.dynamic_ncols = False
        bar.leave = True
        bar.file = sys.stdout
        return bar
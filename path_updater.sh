#path update script
#!/bin/bash

# Fix UNet import
sed -i 's|from physicsnemo.models.diffusion_unets import CorrDiffRegressionUNet|from physicsnemo.models.diffusion.unet import UNet as CorrDiffRegressionUNet|g' train.py

# Fix preconditioning import
sed -i 's|from physicsnemo.diffusion.preconditioners import EDMPrecondSuperResolution|from physicsnemo.models.diffusion.preconditioning import EDMPrecondSuperResolution|g' train.py

# Fix metrics/loss imports
sed -i 's|from physicsnemo.diffusion.metrics import RegressionLoss, ResidualLoss, RegressionLossCE|from physicsnemo.metrics.diffusion.loss import RegressionLoss, ResidualLoss, RegressionLossCE|g' train.py

# Fix patching import
sed -i 's|from physicsnemo.diffusion.multi_diffusion import RandomPatching2D|from physicsnemo.utils.patching import RandomPatching2D|g' train.py

# Fix wandb logging import
sed -i 's|from physicsnemo.utils.logging.wandb import initialize_wandb|from physicsnemo.launch.logging.wandb import initialize_wandb|g' train.py

# Fix console logging imports
sed -i 's|from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper|from physicsnemo.launch.logging.console import PythonLogger, RankZeroLoggingWrapper|g' train.py

# Fix checkpoint utilities import
sed -i 's|from physicsnemo.utils import (|from physicsnemo.launch.utils.checkpoint import (|g' train.py

# Fix InfiniteSampler import (in datasets/dataset.py)
sed -i 's|from physicsnemo.diffusion.utils import InfiniteSampler|from physicsnemo.utils.diffusion.utils import InfiniteSampler|g' datasets/dataset.py

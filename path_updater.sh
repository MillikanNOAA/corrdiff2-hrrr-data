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




# Apply all the import path fixes to generate.py

# Fix logging imports
sed -i 's|from physicsnemo.utils.logging.wandb import initialize_wandb|from physicsnemo.launch.logging.wandb import initialize_wandb|g' generate.py
sed -i 's|from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper|from physicsnemo.launch.logging.console import PythonLogger, RankZeroLoggingWrapper|g' generate.py

# Fix checkpoint imports
sed -i 's|from physicsnemo.utils import (|from physicsnemo.launch.utils.checkpoint import (|g' generate.py

# Fix model imports
sed -i 's|from physicsnemo.models.diffusion_unets.unet import CorrDiffRegressionUNet|from physicsnemo.models.diffusion.unet import UNet as CorrDiffRegressionUNet|g' generate.py

# Fix preconditioning imports
sed -i 's|from physicsnemo.diffusion.preconditioners import EDMPrecondSuperResolution|from physicsnemo.models.diffusion.preconditioning import EDMPrecondSuperResolution|g' generate.py

# Fix metrics imports
sed -i 's|from physicsnemo.diffusion.metrics import|from physicsnemo.metrics.diffusion.loss import|g' generate.py

# Fix patching imports
sed -i 's|from physicsnemo.diffusion.multi_diffusion import RandomPatching2D|from physicsnemo.utils.patching import RandomPatching2D|g' generate.py

# Fix GridPatching2D import
sed -i 's|from physicsnemo.diffusion.multi_diffusion import GridPatching2D|from physicsnemo.utils.patching import GridPatching2D|g' generate.py


# Fix the samplers import (lines 37-40)
sed -i '37,40c\
from physicsnemo.utils.diffusion.deterministic_sampler import deterministic_sampler\
from physicsnemo.utils.diffusion.stochastic_sampler import stochastic_sampler' generate.py

# Fix the generate import (line 41)
sed -i 's|from physicsnemo.diffusion.generate import regression_step, diffusion_step|from physicsnemo.utils.corrdiff.utils import regression_step, diffusion_step|g' generate.py

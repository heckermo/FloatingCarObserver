import os
import sys
import torch
import wandb
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from engine.monocon_engine import MonoconEngine
from utils.engine_utils import tprint, get_default_cfg, set_random_seed, generate_random_seed



# Some Torch Settings
torch_version = int(torch.__version__.split('.')[1])
if torch_version >= 7:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# Get Config from 'config/monocon_configs.py'
cfg = get_default_cfg()


# Set Benchmark
# If this is set to True, it may consume more memory. (Default: True)
if cfg.get('USE_BENCHMARK', True):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    tprint(f"CuDNN Benchmark is enabled.")


# Set Random Seed
seed = cfg.get('SEED', -1)
seed = generate_random_seed(seed)
set_random_seed(seed)

cfg.SEED = seed
tprint(f"Using Random Seed {seed}")

# adapt the ImageSets
# train_files = os.listdir(os.path.join(cfg.DATA.ROOT, 'training', 'calib'))
# train_files = [file.split('.')[0] for file in train_files]
# train_files.sort()
#with open(os.path.join('dataset', 'ImageSets', 'train.txt'), 'w') as f:
#    f.write('')
# with open(os.path.join('dataset', 'ImageSets', 'train.txt'), 'w') as f:
#     for file in train_files:
#         f.write(file + '\n')
# test_files = os.listdir(os.path.join(cfg.DATA.ROOT, 'testing', 'calib'))
# test_files = [file.split('.')[0] for file in test_files]
# test_files.sort()
# #with open(os.path.join('dataset', 'ImageSets', 'trainval.txt'), 'w') as f:
# #    f.write('')
# with open(os.path.join('dataset', 'ImageSets', 'trainval.txt'), 'w') as f:
#     for file in test_files:
#         f.write(file + '\n')
# with open(os.path.join('dataset', 'ImageSets', 'val.txt'), 'w') as f:
#     f.write('')
# with open(os.path.join('dataset', 'ImageSets', 'val.txt'), 'a') as f:
#     for file in test_files:
#         f.write(file + '\n')


# Initialize Engine
engine = MonoconEngine(cfg, auto_resume=False)

# Initialize Weights and Biases
run_name = f'{time.strftime("%Y-%m-%d_%H-%M-%S")}'
wandb.init(project='monocon', name=run_name, config=cfg)


# Start Training from Scratch
# Output files will be saved to 'cfg.OUTPUT_DIR'.

engine.train()
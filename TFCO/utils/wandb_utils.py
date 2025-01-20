import wandb
from typing import Dict

# wandb_config = {
#     'scheduler': {
#         'default': default_scheduler_config,
#         'cosine': cosine_scheduler_config
#     },
#     'config': config,
#     'network_config': masked_transformer_config
# }



def start_wandb(cfg: Dict, network_config: Dict, filename: str, project_name: str, mode: str = 'disabled'):
    while mode not in ["online", "offline", "disabled"]:  
        mode = input('wandb mode (online, offline, disabled): ')
    print(f'wandb mode: {mode}')
    wandb.init(project="tfco", mode=mode, name=filename)
    wandb.config.update(cfg)
    wandb.config.update(network_config['MaskedTransformer'])



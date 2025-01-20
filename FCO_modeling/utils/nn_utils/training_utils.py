import numpy as np

def lr_warmup_decay(warmup_epochs, total_epochs, lr_min=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (
                1
                + np.cos(
                    np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                )
            )

    return lr_lambda
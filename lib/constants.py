from pathlib import Path


def get_hyperparameters():
    hyperparameters = {
        'batch_size': 64,
        'checkpoints_dir': f"{Path.cwd()}/checkpoints",
        'disable_cuda': False,
        'imgs_dir': f"{Path.cwd()}/data/CelebaHQ",
        'labels_filename': f"{Path.cwd()}/data/list_attr_celeba.txt",
        'logs_dir': f"{Path.cwd()}/logs",
        'n_epochs': 10,
        'n_imgs': 30000,
        'verbose': True,
        'version': "0.1"
    }

    return hyperparameters


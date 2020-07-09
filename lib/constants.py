from pathlib import Path


HYPERPARAMETERS = {
    'batch_size': 64,
    'checkpoints_dir': f"{Path.cwd()}/checkpoints",
    'checkpoint_filename': "",
    'disable_cuda': False,
    'imgs_dir': f"{Path.cwd()}/data/CelebaHQ",
    'label': 'Male',
    'labels_filename': f"{Path.cwd()}/data/list_attr_celeba.txt",
    'logs_dir': f"{Path.cwd()}/logs",
    'n_epochs': 10,
    'n_imgs': 30000,
    'verbose': False,
    'version': "0.1",
}
#TODO traintest split


SELECTED_FEATURES = {
    'Bald': 5,
    'Eyeglasses': 16,
    'Male': 21,
    'Smiling': 32,
    'Young': 40,
}


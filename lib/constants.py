from pathlib import Path


HYPERPARAMETERS = {
    'batch_size': 64,
    'disable_cuda': False,
    'label': 'Male',
    'n_epochs': 10,
    'n_imgs': 30000,
    'verbose': False,
    'version': "0.1",
}


PATHS = {
    'checkpoints_dir': f"{Path.cwd()}/checkpoints",
    'checkpoint_filename': "",
    'imgs_dir': f"{Path.cwd()}/data/CelebaHQ",
    'labels_filename': f"{Path.cwd()}/data/labels.txt",
    'logs_dir': f"{Path.cwd()}/logs",
}
#TODO traintest split


SELECTED_FEATURES = {
    'Bald': 5,
    'Eyeglasses': 16,
    'Male': 21,
    'Smiling': 32,
    'Young': 40,
}


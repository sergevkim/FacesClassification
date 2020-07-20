from pathlib import Path


HYPERPARAMETERS = {
    'batch_size': 64,
    'disable_cuda': False,
    'label': 'Male',
    'n_epochs': 10,
    'n_imgs': 30000,
    'test_size': 0.1,
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
    '5_o_Clock_Shadow': 0,
    'Arched_Eyebrows': 1,
    'Attractive': 2,
    'Bags_Under_Eyes': 3,
    'Bald': 4,
    'Bangs': 5,
    'Eyeglasses': 15,
    'Male': 20,
    'Smiling': 31,
    'Young': 39,
}


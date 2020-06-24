from torch.optim import Adam

from lib.models import SimpleClassifier
from lib.utils import get_data_loaders, train_parse_args


class Trainer:
    def __init__(self, mode='.py', params=None):
        self.mode = mode
        self.params = params

    def train(self):
        if mode == '.py':
            params = train_parse_args()
        elif mode == '.ipynb':
            params = self.params
        assert params != None

        model = SimpleClassifier()
        optimizer = Adam(model.parameters(), lr=3e-4, momentum=0.8) #TODO find_lr
        criterion = None #TODO

        loaders = get_data_loaders()
        train_loader = loaders['train_loader']
        val_loader = loaders['val_loader']

        for epoch in range(epochs_num):
            


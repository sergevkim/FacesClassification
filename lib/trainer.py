from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from lib.models import SimpleClassifier
from lib.utils import get_data_loaders, train_parse_args


class Trainer:
    def __init__(self, mode='.py', model=None, optimizer=None, criterion=None, params=None):
        if mode == '.py':
            self.params = train_parse_args()
        elif mode == '.ipynb':
            self.params = params
        assert params != None

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = SummaryWriter(self.params['logs_dir'])


    def save(self, epoch):
        state_dict = {
            'model': self.model,
            'optimizer': self.optimizer,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
        }
        checkpoint_path = "{}/v{}-e{}.hdf5".format(
            self.params['checkpoints_dir'],
            self.params['version'],
            epoch)
        torch.save(state_dict, checkpoint_path)


    def log(self, accuracy, epoch):
        self.model.eval()

        accuracy = validate_classifier(self.model)

        self.writer.add_scalar('accuracy', accuracy.item(), epoch)


    def train_one_epoch(self, train_loader, epoch):
        for batch_idx, batch in enumerate(train_loader):
            self.model.train()
            self.optimizer.zero_grad()

            inputs, labels = batch
            outputs = self.model(inputs)

            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()


    def validation(self, val_loader, epoch):
        self.model.val()

        accuracy = None #TODO

        self.save(epoch)
        self.log(accuracy, epoch)


    def run(self, loaders):
        train_loader = loaders['train_loader']
        valid_loader = loaders['valid_loader']

        for epoch in range(params['n_epochs']):
            if self.params.verbose:
                print("EPOCH {}".format(epoch))

            self.train_one_epoch(train_loader, epoch)
            self.valid(valid_loader, epoch)


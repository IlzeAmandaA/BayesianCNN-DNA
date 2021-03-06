

class EarlyStopping():
    def __init__(self, patience=10):
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.checkpoint = None
        self.checkpoint_name=None


    def is_better(self, metric):
        if self.best is None:
            self.best = metric
            return True
        if metric < self.best:
            self.num_bad_epochs = 0
            self.best = metric
            return True
        else:
            self.num_bad_epochs+=1
            return False

    def store_model(self, ckp, ckp_name):
        self.checkpoint = ckp
        self.checkpoint_name = ckp_name

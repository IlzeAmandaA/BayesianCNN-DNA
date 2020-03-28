

class EarlyStopping():
    def __init__(self, patience=10):
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if self.is_better(metrics):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def is_better(self, metric):
        if metric > self.best:
            return True
        else:
            return False
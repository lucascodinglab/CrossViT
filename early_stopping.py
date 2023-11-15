class EarlyStopping:
    """Early stops the training if validation accuracy doesn't improve after a given patience."""

    def __init__(self, patience=5, delta=1e-3):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        self.val_acc_max = 0.25
        self.update = False

    def check(self, val_acc):
        self.update = False
        if self.val_acc_max >= val_acc:
            self.counter += 1
            print(f'{self.counter} Time(s) no update...')
            if self.counter >= self.patience:
                print('out of patience, early stopped...')
                self.early_stop = True
        else:
            self.update = True
            self.val_acc_max = val_acc
            self.counter = 0
            print('update model!')
        return self.early_stop

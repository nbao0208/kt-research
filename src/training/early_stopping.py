import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_value: float, epoch: int) -> bool:
        if self.mode == "max":
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1
        else:
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1

        if self.counter >= self.patience:
            logger.info(
                "Early stopping triggered after %d epochs without improvement (best: %.6f at epoch %d)",
                self.patience,
                self.best_value,
                self.best_epoch,
            )
            self.early_stop = True

        return self.early_stop

class TrainerLogger:

    def __init__(self):
        self.loss = []
    
    def log(self, loss):
        copied_loss = loss.copy()
        self.loss.append(copied_loss)

    def plot(self):
        len = len(self.loss)
        import matplotlib.pyplot as plt
        plt.plot(range(len), self.loss)
        plt.savefig("./loss.png")
class AverageMeter():
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, count=1):
        self.max = max(val, self.max)
        self.val = val
        self.sum += val * count
        self.count += count
        self.avg = self.sum / self.count

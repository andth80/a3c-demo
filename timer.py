from datetime import datetime


class Timer():

    def __init__(self):
        self.start = datetime.now()
        self.steps = 0

    def split(self, steps):
        end = datetime.now()

        rate = (steps - self.steps) / (end - self.start).total_seconds()

        self.start = end
        self.steps = steps

        return rate

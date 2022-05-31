import copy

class prop:
    def __init__(self, r, c, color, alpha):
        self.r = r
        self.c = c
        self.color = color
        self.alpha = alpha


class Gene:
    def __init__(self):
        self.prop = []

    def copy(self):
        return copy.deepcopy(self)

class MultiFrame:
    """
    Class that represents each frame in different forms of processing

    Attributes:
        rgb:    Original RGB image
        r:      Red values of the original
        g:      Green values of the original
        b:      Blue values of the original
        bw:     Binary threshold image
    """
    rgb = r = g = b = bw = None

    def __init__(self, frame):
        self.rgb = frame
        self.r = frame.copy()
        self.g = frame.copy()
        self.b = frame.copy()
        self.r[:, :, 1] = 0
        self.r[:, :, 0] = 0
        self.g[:, :, 2] = 0
        self.g[:, :, 0] = 0
        self.b[:, :, 1] = 0
        self.b[:, :, 2] = 0

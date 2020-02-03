class Base(object):
    def __init__(self):
        self.base = None

    def setup(self):

        raise NotImplementedError("Must be subclassed.")

    @property
    def get(self):
        if self.base is None:
            self.setup()
#        print(self.base.summary())
        return self.base

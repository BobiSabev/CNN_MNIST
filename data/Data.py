from data.Dataset import Dataset
class Data(object):
    def __init__(self):
        self.train = Dataset(0,0)
        self.validation = Dataset(0,0)
        self.test =Dataset(0,0)


    """

    def __init__(self, train, validate, test):
        self.train = train
        self.validate = validate
        self.test = test
    """
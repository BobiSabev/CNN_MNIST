from data.Dataset import Dataset
from pandas import read_csv
from sklearn.model_selection import train_test_split
class Data(object):
    def __init__(self, train=Dataset(0,0), validation=Dataset(0,0), test=Dataset(0,0)):
        self.train = train
        self.validation = validation
        self.test = test

    # TODO what about 'label', ADAPT FOR TEST !!!!!!, add a loading bar for reading
    def read_data(self, filepath, train_size=2000, validation_size=0, convert_to_one_hot=False):
        print("Reading data...")
        train_data_read = read_csv(filepath, nrows=(train_size + validation_size))
        # Split into the training data into train and validation
        X_train, X_validation, y_train, y_validation = train_test_split(train_data_read.drop(['label'], axis=1),
                                                                        train_data_read['label'],
                                                                        test_size=validation_size / (train_size + validation_size))

        self.train = Dataset(X_train, y_train)
        self.validation = Dataset(X_validation, y_validation)

        # Convert to one-hot vector. Example: 8 -> [ 0.  0.  0.  0.  0.  0.  0.  0.  1.  0.]
        if convert_to_one_hot:
            self.train.to_one_hot()
            self.validation.to_one_hot()

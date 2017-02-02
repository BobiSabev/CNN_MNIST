import numpy as np
import matplotlib.pyplot as plt

class Dataset(object):
    """
        Dataset is a object to hold ndarrays of data and labels

        Attributes:
            images - the data for the network
            labels - target for every image
        Functions
            to_one_hot - converts labels in the object to one hot vectors
            batch - take a random batch of given predefined size
            display_digit - display a random image from the dataset
    """
    def __init__(self, images, labels):
        self.images = np.array(images, dtype=float)
        self.labels = np.array(labels)

    def to_one_hot(self):
        num_classes = np.unique(self.labels).shape[0]
        num_labels = self.labels.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        one_hot_vector = np.zeros((num_labels, num_classes))
        one_hot_vector.flat[index_offset + self.labels.ravel()] = 1
        self.labels = one_hot_vector

    def batch(self, num_rows):
        rows = np.random.randint(self.images.shape[1], size=num_rows)
        return self.images[rows, :], self.labels[rows, :]

    def display_digit(self):
        num = np.random.randint(self.images.shape[1])
        print(self.labels[num])
        label = self.labels[num].argmax(axis=0)
        image = self.images[num].reshape([28, 28])
        plt.title('Example: %d  Label: %d' % (num, label))
        plt.imshow(image, cmap=plt.get_cmap('gray_r'))
        plt.show()



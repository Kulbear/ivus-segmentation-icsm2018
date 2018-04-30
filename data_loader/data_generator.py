import numpy as np


class IVUSDataGenerator():
    def __init__(self, config):
        self.config = config
        # load data here
        img_size = config.state_size[0]
        target = config.target[:3].lower()
        if self.config.use_aug:
            print('Using augmented data...')
            print('Producing {} masks...'.format(config.target))
            self.input = np.load(
                '../data/aug_train_data_{}.npy'.format(img_size))
            self.y = np.load(
                '../data/aug_train_{}_label_{}.npy'.format(target, img_size))
        else:
            print('Using raw data...')
            self.input = np.load(
                '../data/train_data_{}.npy'.format(img_size))
            self.y = np.load(
                '../data/md_train_{}_label_{}.npy'.format(target, img_size))

        # whatever training data we use
        # we need to use these GT to measure the Jaccard index
        self.train_input = np.load(
            '../data/train_data_{}.npy'.format(img_size))
        self.train_y = np.load(
            '../data/md_train_{}_label_{}.npy'.format(target, img_size)).astype(int)
        self.test_input = np.load(
            '../data/test_data_{}.npy'.format(img_size))
        self.test_y = np.load(
            '../data/md_test_{}_label_{}.npy'.format(target, img_size)).astype(int)

    def next_batch(self, batch_size=5):
        idx = np.random.choice(self.input.shape[0], batch_size, replace=False)
        yield self.input[idx], self.y[idx]

    def get_train_subsets(self, subset_size):
        # get train/test subsets is for calculating the Jaccard index
        train_set_size = self.train_input.shape[0]
        train_subsets = [
            (self.train_input[i:i + subset_size], self.train_y[i:i + subset_size])
            for i in range(0, train_set_size, subset_size)
        ]
        return train_subsets

    def get_test_subsets(self, subset_size):
        # pass all test data to a deep model is impossible with my single GPU's memory
        test_set_size = self.test_input.shape[0]
        test_subsets = [
            (self.test_input[i:i + subset_size], self.test_y[i:i + subset_size])
            for i in range(0, test_set_size, subset_size)
        ]
        return test_subsets

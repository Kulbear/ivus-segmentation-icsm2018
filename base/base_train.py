import tensorflow as tf
import os
from tqdm import tqdm
import numpy as np


class BaseTrain:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(),
                             tf.local_variables_initializer())
        self.sess.run(self.init)

        # used for gather results easier
        # epoch, best score, summary
        self.best_test_result = [0, float('Inf'), {}]

    def train(self):
        for cur_epoch in range(
                self.model.cur_epoch_tensor.eval(self.sess),
                self.config.num_epochs + 1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

        # used for gather results easier
        print('Best Result:', self.best_test_result)
        with open('../result.txt', 'a') as f:
            f.write(self.config['exp_name'] + '\n')
            f.write(str(self.best_test_result)+'\n\n')

    def train_epoch(self):
        """
        implement the logic of epoch:
        - loop ever the number of iteration in the config and call teh train step
        - add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

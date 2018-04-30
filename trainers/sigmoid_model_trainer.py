from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import tensorflow as tf


# A function for calculating the Jaccard measure
# By Mehdi Faraji
def Jaccard(Pred, GT):
    # Pred and GT are both binary image
    # Pred is the predicted mask
    # GT is the Ground Truth labels

    # Handle group norm fixed input dimension problem
    if Pred.shape[0] != GT.shape[0]:
        GT = GT[:Pred.shape[0]]

    intersection = np.sum(Pred & GT)
    union = np.sum(Pred | GT)
    Jacc = intersection / union
    return Jacc


class SigmoidTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(SigmoidTrainer, self).__init__(sess, model, data, config,
                                              logger)

    def train_epoch(self):
        ep = self.sess.run(self.model.cur_epoch_tensor)
        print('Epoch:', ep)
        loop = tqdm(range(self.config.num_iter_per_epoch))

        # train and evaluate on the training set
        losses = []
        for it in loop:
            loss = self.train_step()
            losses.append(loss)
        loss = np.mean(losses)
        jacc = self.train_jaccard()

        # evaluate on the test set
        test_loss, test_jacc = self.test_step()

        print('Train Jaccard Index: {}, Train Loss: {}'.format(jacc, loss))
        print('Test Jaccard Index: {}, Test Loss: {}'.format(test_jacc, test_loss))

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {}

        # only save the best model
        if test_loss < self.best_test_result[1]:
            self.best_test_result = [ep, test_loss, {
                'loss': loss,
                'test_loss': test_loss,
                # 'train_jacc': jacc,
                # 'test_jacc': test_jacc,
            }]
            self.model.save(self.sess)
            print(self.best_test_result)

        summaries_dict['training_loss'] = loss
        summaries_dict['test_loss'] = test_loss

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)

    def train_jaccard(self):
        train_subsets = self.data.get_train_subsets(self.config.batch_size)
        all_preds = []
        for subset in train_subsets:
            # TODO: only for making the training faster
            break
            test_x, test_y = subset
            feed_dict = {self.model.x: test_x, self.model.is_training: False}
            prediction = tf.argmax(self.model.logits, axis=-1)
            pred = self.sess.run(prediction, feed_dict=feed_dict)
            all_preds.append(pred)
        # all_preds = np.vstack(all_preds)

        # return Jaccard(all_preds.astype(int), self.data.train_y)
        return .0

    def test_step(self):
        test_losses = []
        test_subsets = self.data.get_test_subsets(self.config.batch_size)
        all_preds = []
        for subset in test_subsets:
            test_x, test_y = subset
            # feed_dict = {self.model.x: test_x, self.model.is_training: False}
            # prediction = tf.argmax(self.model.logits, axis=-1)
            # pred = self.sess.run(prediction, feed_dict=feed_dict)
            # all_preds.append(pred)
            feed_dict = {
                self.model.x: test_x,
                self.model.original_y: test_y,
                self.model.is_training: False
            }
            loss = self.sess.run(self.model.cross_entropy, feed_dict=feed_dict)
            loss = loss * test_x.shape[0]
            test_losses.append(loss)
        # all_preds = np.vstack(all_preds)

        # return np.sum(test_losses) / self.data.test_input.shape[0], Jaccard(
        #     all_preds.astype(int), self.data.test_y)
        return np.sum(test_losses) / self.data.test_input.shape[0], 0

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {
            self.model.x: batch_x,
            self.model.original_y: batch_y,
            self.model.is_training: True
        }
        _, loss = self.sess.run(
            [self.model.train_step, self.model.cross_entropy],
            feed_dict=feed_dict)
        return loss

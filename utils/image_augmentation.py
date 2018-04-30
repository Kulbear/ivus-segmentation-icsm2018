import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa


class ImageAugmentor:
    def __init__(self, images, mode=None, labels=None, end_to_end=True):
        self._source_images = images
        self._source_labels = labels
        self._mode = mode
        self._result_images = [self._source_images]
        self._result_labels = [self._source_labels]
        self._end_to_end = end_to_end

    def generate(self):
        if self._mode == 'SIMPLE':
            self._geometry_augment()
        else:
            # TODO: add other augmentation
            self._geometry_augment()
        if self._end_to_end:
            return np.vstack(self._result_images), np.vstack(
                self._result_labels)
        else:
            return self._result_images, self._result_labels

    def _geometry_augment(self, dropout_prob=0.15):
        # for those need to modify the mask as well
        flip_lr = iaa.Fliplr(1)
        flip_ud = iaa.Flipud(1)
        flip_lrud = iaa.Sequential([flip_lr, flip_ud])
        mask_modified_augs = [flip_lr, flip_ud, flip_lrud]
        for aug in mask_modified_augs:
            self._result_images.append(aug.augment_images(self._source_images))
            self._result_labels.append(aug.augment_images(self._source_labels))

        # for those don't need to modify the mask
        dropout = iaa.Dropout(p=dropout_prob)
        mask_no_modified_augs = [dropout]
        length = len(self._result_images)
        for aug in mask_no_modified_augs:
            for grp_idx in range(length):
                self._result_images.append(
                    aug.augment_images((self._result_images[grp_idx]).astype(np.uint8)))
                self._result_labels.append(self._result_labels[grp_idx])

    def _noise_augment(self, gauss_sigma=0.50, filter_size=(3, 3)):
        gaussian = iaa.GaussianBlur(
            sigma=gauss_sigma)  # TODO: what sigma value is good?
        average = iaa.AverageBlur(k=filter_size)
        median = iaa.MedianBlur(k=filter_size)
        mask_no_modified_augs = [gaussian, average, median]
        for aug in mask_no_modified_augs:
            self._result_images.append(aug.augment_images(self._source_images))
            self._result_labels.append(self._source_labels)


if __name__ == '__main__':
    for tp in ['lum', 'med']:
        for size in [192]:
            images = np.load('./data/train_data_{}.npy'.format(size))
            labels = np.load('./data/md_train_{}_label_{}.npy'.format(tp, size))
            ia = ImageAugmentor(images, labels=labels, mode='SIMPLE')
            images, labels = ia.generate()
            print(images.shape, labels.shape)
            np.save('./data/aug_train_data_{}.npy'.format(size), images)
            np.save('./data/aug_train_{}_label_{}.npy'.format(tp, size), labels)

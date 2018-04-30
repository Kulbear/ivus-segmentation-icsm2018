import os
import cv2
import numpy as np
from skimage import draw, transform

IMAGE_SIZE = (384, 384)
RESCALE_FACTORS = [0.25, 0.5, 1]
DIR = './data'


# https://github.com/scikit-image/scikit-image/issues/1103
def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords,
                                                    vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


class RawDataProcesser(object):
    def __init__(self,
                 image_path='./Training Set/DCM',
                 label_path='./Training Set/LABELS',
                 is_train=True):
        self.image_path = image_path
        self.label_path = label_path
        self.images = sorted(os.listdir(path=image_path))  # image names list
        self.is_train = is_train

    def _process_label(self, label_type="", img_name=""):
        folder_path = 'Training Set' if self.is_train else 'Test Set'
        label_path = "./{}/LABELS/{}_{}.txt".format(folder_path, label_type,
                                                    img_name)
        with open(label_path) as f:
            cnt = f.read().splitlines()
            coord = [(xy.split(', ')) for xy in cnt]
            x, y = zip(*coord)
        x, y = np.array(x, dtype=np.float_), np.array(y, dtype=np.float_)
        return x, y

    def process_data(self, rescale_factor=1):
        w, h = int(IMAGE_SIZE[0] * rescale_factor), int(
            IMAGE_SIZE[1] * rescale_factor)
        self.img_feature = np.zeros((len(self.images), w, h, 1))
        self.lum_label = np.zeros((len(self.images), w, h))
        self.med_label = np.zeros((len(self.images), w, h))
        with open('processing_order_{}.txt'.format('train' if self.is_train else 'test'), 'w') as f1:
            for idx in range(len(self.images)):
                fname = self.images[idx]
                img_name = fname[:-4]

                # Read and process image pixels
                # TODO: change the way concatenate the file path
                img_fpath = "{}/{}".format(self.image_path, fname)
                f1.write(img_fpath + '\n')
                img = cv2.imread(img_fpath, cv2.IMREAD_GRAYSCALE)
                img = np.array(img, dtype=np.uint8)
                img = transform.rescale(img, rescale_factor)
                img = np.expand_dims(
                    img, axis=-1)  # expand shape to (*IMAGE_SIZE, 1)
                self.img_feature[idx] = img

                # Read and process lumen coordinate labels
                x_lum, y_lum = self._process_label(
                    label_type='lum', img_name=img_name)

                # Read and process media coordinate labels
                x_med, y_med = self._process_label(
                    label_type='med', img_name=img_name)

                # Build lumen mask and media mask
                lum_mask = poly2mask(x_lum, y_lum, IMAGE_SIZE).T
                lum_mask = transform.rescale(lum_mask, rescale_factor).astype(int)
                med_mask = poly2mask(x_med, y_med, IMAGE_SIZE).T
                med_mask = transform.rescale(med_mask, rescale_factor).astype(int)
                self.lum_label[idx] = lum_mask
                self.med_label[idx] = np.bitwise_xor(med_mask, lum_mask)
            # med_mask = transform.rescale(med_mask, rescale_factor).astype(int) * 2
            # self.lum_label[idx] = lum_mask 
            # self.med_label[idx] = med_mask - lum_mask
              # exclude the lum part from the med

    def save(self, rescale_factor=1):  # assume square images for now
        try:  # create the DIR if not existeds
            os.stat(DIR)
        except:
            os.mkdir(DIR)
        # Save processed data to file
        size = int(IMAGE_SIZE[0] * rescale_factor)
        tp = 'train' if self.is_train else 'test'
        np.save('{}/{}_data_{}.npy'.format(DIR, tp, size), self.img_feature)
        np.save('{}/{}_lum_label_{}.npy'.format(DIR, tp, size), self.lum_label)
        np.save('{}/{}_med_label_{}.npy'.format(DIR, tp, size), self.med_label)


if __name__ == '__main__':
    print('Processing training images..')
    processor = RawDataProcesser(
        image_path='./Training Set/DCM', label_path='./Training Set/LABELS')
    for rfactor in RESCALE_FACTORS:
        print('Generating training images with a rescale level {}...'.format(
            rfactor))
        processor.process_data(rescale_factor=rfactor)
        processor.save(rescale_factor=rfactor)

    print('Processing test images..')
    processor = RawDataProcesser(
        is_train=False,
        image_path='./Test Set/DCM',
        label_path='./Test Set/LABELS')
    for rfactor in RESCALE_FACTORS:
        print('Generating test images with a rescale level {}...'.format(
            rfactor))
        processor.process_data(rescale_factor=rfactor)
        processor.save(rescale_factor=rfactor)

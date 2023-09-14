# Dataset utilities for segmentation task


import tensorflow as tf
import os
from pathlib import Path
import keras_cv


class SegmentationDataset():
    def __init__(self, images_dir, masks_dir, augment=False, batch_size=1, shuffle=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.mask_paths = [self.masks_dir+mask_name for mask_name in sorted(os.listdir(self.masks_dir))]
        image_names = sorted([f'{Path(mask_path).stem}.jpg' for mask_path in self.mask_paths]) # Images directory contains all train and validation images, while the masks directory is separated
        self.image_paths = [self.images_dir+image_name for image_name in image_names]

        if self.augment:
            self.augmenter = keras_cv.layers.RandAugment(value_range=(0,1), geometric=False) # Geometric transforms with RandAugment can't be applied identically to images and masks at the moment

    def get_dataset(self):
        seg_dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.mask_paths))
        seg_dataset = seg_dataset.map(self.prepare_images, num_parallel_calls=tf.data.AUTOTUNE)
        seg_dataset = seg_dataset.cache() # perform time intensive mapping before and memory intensive mapping after
        if self.shuffle:
            seg_dataset = seg_dataset.shuffle(buffer_size=seg_dataset.cardinality(), reshuffle_each_iteration=True) # If caching this should be after the cache call, otherwise it's nice to place it before opening the images as the entire dataset of string paths can be shuffled without memory issues
        seg_dataset = seg_dataset.batch(self.batch_size) # try batching before augmenting to vectorize the augmentation - doesn't actually seem to help all that much
        if self.augment:
            seg_dataset = seg_dataset.map(self.augment_images, num_parallel_calls=tf.data.AUTOTUNE)
        # seg_dataset = seg_dataset.batch(self.batch_size) # drop_remainder
        # seg_dataset = seg_dataset.repeat() # Doesn't seem to be needed - I think maybe newer tf/keras code with model.fit handles this automatically
        seg_dataset = seg_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return seg_dataset
        

    def prepare_images(self, image_string_tensor, mask_string_tensor):
        # tf.print(image_string_tensor)
        img = tf.io.read_file(image_string_tensor)
        img = tf.io.decode_jpeg(img)
        img = tf.image.resize(img, [512,512])
        img = img / 255.0 # rescale images

        mask = tf.io.read_file(mask_string_tensor)
        mask = tf.io.decode_png(mask)
        mask = tf.image.resize(mask, [512,512], method='nearest') # Need to be careful with resizing here, since the values should be integers corresponding to classes, we don't want fractional values

        return img, mask
    
    def augment_images(self, image_tensor, mask_tensor):
        img = self.augmenter(image_tensor)
        mask = mask_tensor

        if tf.random.uniform(shape=()) > 0.5:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)

        if tf.random.uniform(shape=()) > 0.5:
            img = tf.image.flip_up_down(img)
            mask = tf.image.flip_up_down(mask)

        return img, mask
   




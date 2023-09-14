# Train a segmentation model.
# This script should be run from the root project directory (same folder as train.py)


import tensorflow as tf
import tf_data_utils
import model_utils
# import os
# import numpy as np
import time

def main():

    # Parameters would be better passed in via argparse
    num_epochs = 20
    batch_size = 8 # Didn't change much when I upped it, definitely places to optimize the training pipeline to be found

    print(f'\nTraining for {num_epochs} epochs with batch size {batch_size}.\n')

    # images_dir = '/Users/NickPC/Documents/DataScience/Datasets/miccai2022_sisvse_dataset/images/real/'
    # train_masks_dir = '/Users/NickPC/Documents/DataScience/Datasets/miccai2022_sisvse_dataset/semantic_masks/real_train_1/'
    # val_masks_dir = '/Users/NickPC/Documents/DataScience/Datasets/miccai2022_sisvse_dataset/semantic_masks/real_val_1/'

    images_dir = '/home/ec2-user/Data/miccai2022_sisvse_dataset/images/real/'
    train_masks_dir = '/home/ec2-user/Data/miccai2022_sisvse_dataset/semantic_masks/real_train_1/'
    val_masks_dir = '/home/ec2-user/Data/miccai2022_sisvse_dataset/semantic_masks/real_val_1/'

    train_dataset = tf_data_utils.SegmentationDataset(images_dir=images_dir, masks_dir=train_masks_dir, augment=True, batch_size=batch_size, shuffle=True).get_dataset()
    val_dataset = tf_data_utils.SegmentationDataset(images_dir=images_dir, masks_dir=val_masks_dir, augment=False, batch_size=1, shuffle=False).get_dataset()

    # temp_train_ds = train_dataset.take(1)
    # temp_val_ds = val_dataset.take(1)

    segmentation_model = model_utils.get_model(img_size=(512,512), num_classes=32)
    # print(segmentation_model.summary())

    segmentation_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"], # There is definitely a better metric to use here
    )

    model_name = "test_segmentation_model"
    # model_name = "test_shuffle"

    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=f'logs/{model_name}')]
    # callbacks = []

    start = time.time()

    segmentation_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=num_epochs,
        callbacks=callbacks
    )

    end = time.time()

    print(f'\n{(end-start)/60.:.3f} minutes to fit.')

    # segmentation_model.fit(
    #     temp_train_ds,
    #     validation_data=temp_val_ds,
    #     epochs=num_epochs,
    #     callbacks=callbacks
    # )

    segmentation_model.save(f'models/{model_name}.keras')

    return


if __name__ == '__main__':
    main()

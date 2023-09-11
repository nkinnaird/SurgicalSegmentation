# Train a segmentation model.


import tensorflow as tf
import tf_data_utils
import model_utils
# import os
# import numpy as np

def main():

    num_epochs = 2
    batch_size = 1

    images_dir = '/Users/NickPC/Documents/DataScience/Datasets/miccai2022_sisvse_dataset/images/real/'
    train_masks_dir = '/Users/NickPC/Documents/DataScience/Datasets/miccai2022_sisvse_dataset/semantic_masks/real_train_1/'
    val_masks_dir = '/Users/NickPC/Documents/DataScience/Datasets/miccai2022_sisvse_dataset/semantic_masks/real_val_1/'


    train_dataset = tf_data_utils.SegmentationDataset(images_dir=images_dir, masks_dir=train_masks_dir, augment=True, batch_size=batch_size, shuffle=False).get_dataset()
    val_dataset = tf_data_utils.SegmentationDataset(images_dir=images_dir, masks_dir=val_masks_dir, augment=False, batch_size=1, shuffle=False).get_dataset()

    temp_train_ds = train_dataset.take(1)
    temp_val_ds = val_dataset.take(1)

    # print(len(temp_train_ds))
    # print(len(temp_val_ds))


    segmentation_model = model_utils.get_model(img_size=(512,512), num_classes=32)
    # print(segmentation_model.summary())

    segmentation_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = segmentation_model.fit(
        temp_train_ds,
        validation_data=temp_val_ds,
        epochs=num_epochs
        # batch_size=batch_size,
        # validation_batch_size=1
        # callbacks=callbacks,
    )






if __name__ == '__main__':
    main()

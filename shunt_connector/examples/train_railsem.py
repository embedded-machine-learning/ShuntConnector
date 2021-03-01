
from pathlib import Path
import sys
import configparser

import tensorflow as tf

from shunt_connector import ShuntConnector
from shunt_connector.utils.dataset_utils import cityscapes_preprocess_image_and_label

def create_railsem_dataset(file_path, input_shape, used_idx, is_training=True):
    if not isinstance(file_path, Path):     # convert str to Path
        file_path = Path(file_path)

    record_file_list = list(map(str, file_path.glob("*")))

    record_file_list_filterd = []

    # filter list into train and val
    for record_name in record_file_list:
        name_as_path = Path(record_name)
        record_idx = int(name_as_path.stem[7:12])
        if record_idx in used_idx:
            record_file_list_filterd.append(record_name)

    def parse_function(example_proto):
        def _decode_image(content, channels):
            return tf.cond(
                tf.image.is_jpeg(content),
                lambda: tf.image.decode_jpeg(content, channels),
                lambda: tf.image.decode_png(content, channels))

        features = {
            'image/encoded':
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/filename':
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
                tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height':
                tf.io.FixedLenFeature((), tf.int64, default_value=0),
            'image/width':
                tf.io.FixedLenFeature((), tf.int64, default_value=0),
            'image/segmentation/class/encoded':
                tf.io.FixedLenFeature((), tf.string, default_value=''),
            'image/segmentation/class/format':
                tf.io.FixedLenFeature((), tf.string, default_value='png'),
        }

        parsed_features = tf.io.parse_single_example(example_proto, features)

        image = _decode_image(parsed_features['image/encoded'], channels=3)

        label = _decode_image(parsed_features['image/segmentation/class/encoded'], channels=1)

        image_name = parsed_features['image/filename']
        if image_name is None:
            image_name = tf.constant('')

        if label is not None:
            if label.get_shape().ndims == 2:
                label = tf.expand_dims(label, 2)
            elif label.get_shape().ndims == 3 and label.shape.dims[2] == 1:
                pass
            else:
                raise ValueError('Input label shape must be [height, width], or '
                            '[height, width, 1].')

        label.set_shape([None, None, 1])
        image, label = cityscapes_preprocess_image_and_label(image,
                                                             label,
                                                             crop_height=input_shape[0],
                                                             crop_width=input_shape[1],
                                                             min_resize_value=input_shape[0],
                                                             max_resize_value=input_shape[1],
                                                             is_training=is_training)

        return image, label

    ds = tf.data.TFRecordDataset(record_file_list_filterd, num_parallel_reads=tf.data.experimental.AUTOTUNE) \
         .map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
        ds = ds.shuffle(100)
        ds = ds.repeat()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


config_path = Path(sys.path[0], "config", "train_railsem.cfg")
config = configparser.ConfigParser()
config.read(config_path)

connector = ShuntConnector(config)

# custom create_dataset()
dataset_path = config['DATASET'].get('path')
TRAIN_IDX = [*range(0,14,1)]
VAL_IDX = [14,15,16]
input_shape = tuple(map(int, config['DATASET']['input_size'].split(',')))

connector.dataset_props['num_classes'] = 19
connector.dataset_props['input_shape'] = (connector.dataset_params['input_size'][0],
                                          connector.dataset_params['input_size'][1],
                                          3)
connector.dataset_props['len_train_data'] = len(TRAIN_IDX) * 8500//20
connector.dataset_props['len_val_data'] = len(VAL_IDX) * 8500//20
connector.dataset_props['task'] = 'segmentation'
connector.test_batchsize = connector.dataset_params['test_batchsize'] # this property is set during the _parse_config call. Check the core/_parse_config file for further fields, which may be useful for you.
connector.load_task_losses_metrics()

connector.dataset_train = create_railsem_dataset(dataset_path, input_shape, TRAIN_IDX, is_training=True)
connector.dataset_val = create_railsem_dataset(dataset_path, input_shape, VAL_IDX, is_training=False)

# now continue with usual procedure
connector.create_original_model()
connector.train_original_model()
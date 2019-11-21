import tensorflow as tf
from tensorflow.contrib.data import Dataset, Iterator
from random import shuffle

DB_PATH = './bird_dataset/'
NUM_CLASSES = 100


class DatasetLoader(object):
    """ Class that loads the provided birds dataset into Tensorflow. """
    def __init__(self, is_train, batch_size=1, img_size=227):
        self._is_train = is_train
        self._batch_size = batch_size
        self._img_size = img_size

        self.get_data = None
        self.num_samples = None
        self.num_classes = NUM_CLASSES

        self._img_list = list()
        self._lable_list = list()
        self._init_op = None

        self._class_id2name_map = dict()

        self._load_data()

    def _load_data(self):
        # Read txt files containing infos about the dataset
        with open(DB_PATH + 'classes.txt', 'r') as fi:
            for line in fi:
                items = line.split(', ')
                self._class_id2name_map[int(items[0])] = items[1].rstrip()

        is_train_list = list()
        with open(DB_PATH + 'train_test_split.txt', 'r') as fi:
            for line in fi:
                items = line.split(', ')
                is_train_list.append(int(items[1]))

        image_list = list()
        with open(DB_PATH + 'images.txt', 'r') as fi:
            for line in fi:
                items = line.split(', ')
                image_list.append('images/' + items[1][:-1])

        class_lables = list()
        with open(DB_PATH + 'image_class_labels.txt', 'r') as fi:
            for line in fi:
                items = line.split(' ')
                class_lables.append(int(items[1])-1)

        # check inputs
        assert len(is_train_list) == len(image_list), "Lengths dont match. Input data seems broken."
        assert len(is_train_list) == len(class_lables), "Lengths dont match. Input data seems broken."

        # shuffle lists
        ind = list(range(len(is_train_list)))
        shuffle(ind)
        is_train_list = [is_train_list[i] for i in ind]
        image_list = [image_list[i] for i in ind]
        class_lables = [class_lables[i] for i in ind]

        # get samples of the set we want
        for is_train, class_lable, file_path in zip(is_train_list, class_lables, image_list):
            if self._is_train and (is_train == 1):
                self._img_list.append(file_path)
                self._lable_list.append(class_lable)

            elif (not self._is_train) and (is_train == 0):
                self._img_list.append(file_path)
                self._lable_list.append(class_lable)
        self.num_samples = len(self._img_list)

        # turn into tensorflow objects
        dataset = Dataset.from_tensor_slices((self._img_list, self._lable_list))

        # load images and turn into one hot encoding
        if self._is_train:
            dataset = dataset.map(self._input_parser_train, num_threads=4)
        else:
            dataset = dataset.map(self._input_parser_test, num_threads=4)

        # shuffle
        dataset = dataset.shuffle(buffer_size=32)

        # repeat indefinitely
        dataset = dataset.repeat(-1)

        # batch the data
        dataset = dataset.batch(self._batch_size)

        # shared iterator
        iterator = Iterator.from_structure(dataset.output_types, dataset.output_shapes)

        # create two initialization ops to switch between the datasets
        self._init_op = iterator.make_initializer(dataset)

        self.get_data = iterator.get_next()

    def init(self, session):
        session.run(self._init_op)

    def _input_parser_train(self, img_path, label):
        # convert the label to one-hot encoding
        one_hot = tf.one_hot(label, NUM_CLASSES)

        # read the img from file
        img_file = tf.read_file(DB_PATH + img_path)
        img_decoded = tf.image.decode_image(img_file, channels=3)
        img_decoded = tf.image.resize_image_with_crop_or_pad(img_decoded, self._img_size, self._img_size)

        return img_decoded, one_hot

    def _input_parser_test(self, img_path, label):
        # convert the label to one-hot encoding
        one_hot = tf.one_hot(label, NUM_CLASSES)

        # read the img from file
        img_file = tf.read_file(DB_PATH + img_path)
        img_decoded = tf.image.decode_image(img_file, channels=3)
        img_decoded = tf.image.resize_image_with_crop_or_pad(img_decoded, self._img_size, self._img_size)

        return img_decoded, one_hot




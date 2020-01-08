""" Retrains AlexNet on a bird classification dataset. """
import tensorflow as tf
import shutil
import os

from alexnet import AlexNet
from dataset_loader import *


# PARAMETERS
IMG_SIZE = 227  # input image size for alexnet
BATCH_SIZE = 10  # number of samples per iteration
TRAIN_ITER = 10000  # number of iterations for training
SNAPSHOT_DIR = './snapshots'  # where the learned weights are saved to

# setting up the data loader
db_train = DatasetLoader(is_train=True, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
db_test = DatasetLoader(is_train=False, batch_size=BATCH_SIZE, img_size=IMG_SIZE)

# setup placeholder inputs
image_tf = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3), name='image')
class_prob_gt_tf = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, db_train.num_classes), name='class_prob')

# network
net = AlexNet(keep_prob=0.5, num_classes=db_train.num_classes)
_, scores = net.inference(image_tf)  # sets up the network architecture and returns predicted unnormalized class scores



##
# TODO: DEFINE TRAINING LOSS AND OPTIMIZER HERE
loss_v = tf.nn.softmax_cross_entropy_with_logits(labels=class_prob_gt_tf, logits=scores)
loss = tf.reduce_mean(loss_v)
opt_gd = tf.train.GradientDescentOptimizer(learning_rate=1e-4)

train_step = opt_gd.minimize(loss)
##


# start TF session and init data loaders
sess = tf.Session()
db_train.init(sess)
db_test.init(sess)

# initialize network weights
sess.run(tf.global_variables_initializer())  # this initialized the weights randomly
net.load_initial_weights(sess)  # this loads the pretrained alexnet weights from the npy file

# saver for the network weights
saver = tf.train.Saver(max_to_keep=1)
if not os.path.exists(SNAPSHOT_DIR):
    os.mkdir(SNAPSHOT_DIR)
else:
    shutil.rmtree(SNAPSHOT_DIR)
    os.mkdir(SNAPSHOT_DIR)

# TRAINING LOOP
tr_loss = []
for step in range(TRAIN_ITER):
    # work around to deal with cases when batch size doesnt divide the number of samples of the db set evenly
    while True:
        # load images and annotation
        img_v, class_gt_v = sess.run(db_train.get_data)
        if img_v.shape[0] == BATCH_SIZE:
            break

    # TODO: training iteration
    _, loss_value = sess.run([train_step,loss], feed_dict={image_tf: img_v, class_prob_gt_tf: class_gt_v})
    if step%100 == 0:
      tr_loss += [loss_value]
      print('STEP', step ,' -------- LOSS', loss_value)

print('Training finished. Saving final snapshot.')
saver.save(sess, "%s/model" % SNAPSHOT_DIR, global_step=TRAIN_ITER)


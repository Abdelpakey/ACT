from __future__ import division
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from scipy.misc import imresize
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from src.init_paras import paras_init
import skimage
import skimage.io
import skimage.transform
from data_prov import *


VGG_MEAN = [128, 128, 128]
critic_list = ['conv1', 'conv2', 'conv3','conv4', 'fc1', 'fc2']
actor_list = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2']

############  hyper parameters  ####################
raidus = 5
alpha = 0.0001
bias = 2
beta = 0.75
LR_A_int = 0.0001
LR_C_int = 0.0001
LR_A = 0.000001  # learning rate for actor
LR_C = 0.00001  # learning rate for critic
GAMMA = 0.99  # reward discount
REPLACE_ITER_A = 10000
REPLACE_ITER_C = 8000
BATCH_SIZE = 128
TAU = 0.001
cropsize = 1
MEMORY_CAPACITY = 10000

class DDPG(object):
    def __init__(self, sess, actor_npy_path, critic_npy_path):
        self.pointer = 0
        self.sess = sess
        self.a_replace_counter, self.c_replace_counter = 0, 0
        self.a_replace_counter_init, self.c_replace_counter_init = 0, 0

        self.memory_sg = np.zeros(MEMORY_CAPACITY*107*107*3).reshape(MEMORY_CAPACITY, 107, 107, 3)
        self.memory_sg_ = np.zeros(MEMORY_CAPACITY * 107 * 107 * 3).reshape(MEMORY_CAPACITY, 107, 107, 3)
        self.memory_a = np.zeros(MEMORY_CAPACITY * 3).reshape(MEMORY_CAPACITY, 3)
        self.memory_r = np.zeros(MEMORY_CAPACITY * 1).reshape(MEMORY_CAPACITY, 1)
        self.memory_sl = np.zeros(MEMORY_CAPACITY*107*107*3).reshape(MEMORY_CAPACITY, 107, 107, 3)
        self.memory_sl_ = np.zeros(MEMORY_CAPACITY * 107 * 107 * 3).reshape(MEMORY_CAPACITY, 107, 107, 3)

        # self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.Sl = tf.placeholder(tf.float32, [None, 107, 107, 3], 'sl')
        self.Sg = tf.placeholder(tf.float32, [None, 107, 107, 3], 'sg')
        self.Sl_ = tf.placeholder(tf.float32, [None, 107, 107, 3], 'sl_')
        self.Sg_ = tf.placeholder(tf.float32, [None, 107, 107, 3], 'sg_')

        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.target_q = tf.placeholder(tf.float32, [None, 1], 'target_q')
        self.actor_true_out = tf.placeholder(tf.float32, [None, 3], 'a')

        with tf.variable_scope('Actor'):
            self.a = self._build_actor(self.Sg, self.Sl, actor_npy_path, scope='eval')
            self.a_ = self._build_actor(self.Sg_, self.Sl_, actor_npy_path, scope='target')
        with tf.variable_scope('Critic'):
            self.q = self._build_critic(self.Sg, self.Sl, self.a, critic_npy_path, scope='eval')
            self.q_ = self._build_critic(self.Sg_, self.Sl_,  self.a_, critic_npy_path, scope='target')

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor' + '/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor' + '/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic' + '/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic' + '/target')

        q_target = self.R + GAMMA * self.q_

        self.td_error = tf.reduce_mean(tf.squared_difference(q_target, self.q))
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.td_error, var_list=self.ce_params)

        self.actor_loss = tf.reduce_mean(tf.squared_difference(self.a, self.actor_true_out))
        self.actor_loss_ = tf.reduce_mean(tf.squared_difference(self.a_, self.actor_true_out))
        self.train_actor = tf.train.AdamOptimizer(LR_A_int).minimize(self.actor_loss, var_list=self.ae_params)
        self.train_actor_ = tf.train.AdamOptimizer(LR_A_int).minimize(self.actor_loss_, var_list=self.at_params)

        self.action_grads = tf.gradients(self.q, self.a)
        self.policy_grads = tf.gradients(ys=self.a, xs=self.ae_params, grad_ys=self.action_grads[0])

        opt = tf.train.AdamOptimizer(-LR_A)
        self.atrain = opt.apply_gradients(zip(self.policy_grads, self.ae_params))

        self.update_actor_params = \
            [self.at_params[i].assign(tf.multiply(self.ae_params[i], TAU) +
                                   tf.multiply(self.at_params[i], 1. - TAU))
             for i in range(len(self.at_params))]

        self.update_critic_params = \
            [self.ct_params[i].assign(tf.multiply(self.ce_params[i], TAU) +
                                   tf.multiply(self.ct_params[i], 1. - TAU))
             for i in range(len(self.ct_params))]

    def take_up(self):
        self.sess.run([tf.assign(t, e) for t, e in zip(self.at_params, self.ae_params)])
        self.sess.run([tf.assign(t, e) for t, e in zip(self.ct_params, self.ce_params)])

    def learn(self):
        self.idx = np.random.choice(min(self.pointer, MEMORY_CAPACITY), BATCH_SIZE)
        bsg = self.memory_sg[self.idx]
        bsl = self.memory_sl[self.idx]
        ba = self.memory_a[self.idx]
        br = self.memory_r[self.idx]
        bsg_ = self.memory_sg_[self.idx]
        bsl_ = self.memory_sl_[self.idx]
        self.sess.run(self.ctrain, {self.Sg: bsg, self.Sl: bsl, self.a: ba, self.R: br, self.Sg_:bsg_, self.Sl_:bsl_})
        self.sess.run(self.atrain, {self.Sg: bsg, self.Sl: bsl})

        self.a_replace_counter += 1; self.c_replace_counter += 1

        if self.a_replace_counter % REPLACE_ITER_A == 0:
            self.sess.run(self.update_actor_params)
        if self.c_replace_counter % REPLACE_ITER_C == 0:
            self.sess.run(self.update_critic_params)

    def show_critic_loss(self):
        bsg = self.memory_sg[self.idx]
        bsg_ = self.memory_sg_[self.idx]
        ba = self.memory_a[self.idx]
        br = self.memory_r[self.idx]
        bsl = self.memory_sl[self.idx]
        bsl_ = self.memory_sl_[self.idx]

        return self.sess.run(self.td_error, {self.Sl: bsl, self.Sg: bsg, self.Sg_: bsg_, self.a: ba, self.R: br, self.Sl_: bsl_})


    def store_transition(self, sg, sl, a, r, sg_, sl_):
        index = self.pointer % MEMORY_CAPACITY
        self.memory_sg[index] = sg
        self.memory_sl[index] = sl
        self.memory_a[index] = a
        self.memory_r[index] = r
        self.memory_sg_[index] = sg_
        self.memory_sl_[index] = sl_
        self.pointer += 1

    def init_actor(self, sg, sl, a):
        self.sess.run(self.train_actor, {self.Sl: sl, self.Sg: sg, self.actor_true_out: a})
        self.sess.run(self.train_actor_, {self.Sl_: sl, self.Sg_: sg, self.actor_true_out: a})

    def _build_actor(self, rgb_g, rgb_l, actor_npy_path, scope):
        if actor_npy_path is not None:
            self.data_dict = np.load(actor_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None
        rgb_g = rgb_g - 128.0
        rgb_l = rgb_l - 128.0
        with tf.variable_scope(scope):
            conv1w = tf.Variable(self.data_dict['conv1'][0], name='conv1_filters')
            conv1b = tf.Variable(self.data_dict['conv1'][1], name='conv1_biases')
            conv2w = tf.Variable(self.data_dict['conv2'][0], name='conv2_filters')
            conv2b = tf.Variable(self.data_dict['conv2'][1], name='conv2_biases')
            conv3w = tf.Variable(self.data_dict['conv3'][0], name='conv3_filters')
            conv3b = tf.Variable(self.data_dict['conv3'][1], name='conv3_biases')
            conv4w = tf.Variable(self.data_dict['conv4'][0], name='conv4_filters')
            conv4b = tf.Variable(self.data_dict['conv4'][1], name='conv4_biases')

            conv1_g = tf.nn.conv2d(rgb_g, conv1w, [1, 2, 2, 1], padding='VALID')
            bias1_g = tf.nn.bias_add(conv1_g, conv1b)
            relu1_g = tf.nn.relu(bias1_g)
            lrn1_g = tf.nn.local_response_normalization(relu1_g, depth_radius=raidus, alpha=alpha, beta=beta,
                                                        bias=bias)
            pool1_g = self.max_pool(lrn1_g, 'pool1_l')

            conv2_g = tf.nn.conv2d(pool1_g, conv2w, [1, 2, 2, 1], padding='VALID')
            bias2_g = tf.nn.bias_add(conv2_g, conv2b)
            relu2_g = tf.nn.relu(bias2_g)
            lrn2_g = tf.nn.local_response_normalization(relu2_g, depth_radius=raidus, alpha=alpha, beta=beta,
                                                        bias=bias)
            pool2_g = self.max_pool(lrn2_g, 'pool2_l')

            conv3_g = tf.nn.conv2d(pool2_g, conv3w, [1, 1, 1, 1], padding='VALID')
            bias3_g = tf.nn.bias_add(conv3_g, conv3b)
            relu3_g = tf.nn.relu(bias3_g)

            conv4_g = tf.nn.conv2d(relu3_g, conv4w, [1, 1, 1, 1], padding='VALID')
            bias4_g = tf.nn.bias_add(conv4_g, conv4b)
            relu4_g = tf.nn.relu(bias4_g)

            conv1_l = tf.nn.conv2d(rgb_l, conv1w, [1, 2, 2, 1], padding='VALID')
            bias1_l = tf.nn.bias_add(conv1_l, conv1b)
            relu1_l = tf.nn.relu(bias1_l)
            lrn1_l = tf.nn.local_response_normalization(relu1_l, depth_radius=raidus, alpha=alpha, beta=beta,
                                                        bias=bias)
            pool1_l = self.max_pool(lrn1_l, 'pool1_l')

            conv2_l = tf.nn.conv2d(pool1_l, conv2w, [1, 2, 2, 1], padding='VALID')
            bias2_l = tf.nn.bias_add(conv2_l, conv2b)
            relu2_l = tf.nn.relu(bias2_l)
            lrn2_l = tf.nn.local_response_normalization(relu2_l, depth_radius=raidus, alpha=alpha, beta=beta,
                                                        bias=bias)
            pool2_l = self.max_pool(lrn2_l, 'pool2_l')

            conv3_l = tf.nn.conv2d(pool2_l, conv3w, [1, 1, 1, 1], padding='VALID')
            bias3_l = tf.nn.bias_add(conv3_l, conv3b)
            relu3_l = tf.nn.relu(bias3_l)

            conv4_l = tf.nn.conv2d(relu3_l, conv4w, [1, 1, 1, 1], padding='VALID')
            bias4_l = tf.nn.bias_add(conv4_l, conv4b)
            relu4_l = tf.nn.relu(bias4_l)

            fc_input1 = tf.reshape(relu4_g, [-1, 512])
            fc_input2 = tf.reshape(relu4_l, [-1, 512])

            fc_input = tf.concat([fc_input1, fc_input2], 1)


            fc1 = self.fc_layer(fc_input, 1024, 512, 'fc1')
            relu4 = tf.nn.relu(fc1, 'relu4')
            fc2 = self.fc_layer(relu4, 512, 3, 'fc2')
            out = tf.nn.tanh(fc2, name='out')
            # out[:,2] = tf.clip_by_value(out , -0.05, 0.05)
            self.data_dict = None
        return out


    def _build_critic(self, rgb_g, rgb_l, a, critic_npy_path, scope):
        if critic_npy_path is not None:
            self.data_dict = np.load(critic_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        rgb_g = rgb_g - 128.0
        rgb_l = rgb_l - 128.0
        with tf.variable_scope(scope):
            conv1w = tf.Variable(self.data_dict['conv1'][0], name='conv1_filters')
            conv1b = tf.Variable(self.data_dict['conv1'][1], name='conv1_biases')
            conv2w = tf.Variable(self.data_dict['conv2'][0], name='conv2_filters')
            conv2b = tf.Variable(self.data_dict['conv2'][1], name='conv2_biases')
            conv3w = tf.Variable(self.data_dict['conv3'][0], name='conv3_filters')
            conv3b = tf.Variable(self.data_dict['conv3'][1], name='conv3_biases')
            conv4w = tf.Variable(self.data_dict['conv4'][0], name='conv4_filters')
            conv4b = tf.Variable(self.data_dict['conv4'][1], name='conv4_biases')

            conv1_g = tf.nn.conv2d(rgb_g, conv1w, [1, 2, 2, 1], padding='VALID')
            bias1_g = tf.nn.bias_add(conv1_g, conv1b)
            relu1_g = tf.nn.relu(bias1_g)
            lrn1_g = tf.nn.local_response_normalization(relu1_g, depth_radius=raidus, alpha=alpha, beta=beta,
                                                           bias=bias)
            pool1_g = self.max_pool(lrn1_g, 'pool1_l')

            conv2_g = tf.nn.conv2d(pool1_g, conv2w, [1, 2, 2, 1], padding='VALID')
            bias2_g = tf.nn.bias_add(conv2_g, conv2b)
            relu2_g = tf.nn.relu(bias2_g)
            lrn2_g = tf.nn.local_response_normalization(relu2_g, depth_radius=raidus, alpha=alpha, beta=beta,
                                                           bias=bias)
            pool2_g = self.max_pool(lrn2_g, 'pool2_l')

            conv3_g = tf.nn.conv2d(pool2_g, conv3w, [1, 1, 1, 1], padding='VALID')
            bias3_g = tf.nn.bias_add(conv3_g, conv3b)
            relu3_g = tf.nn.relu(bias3_g)

            conv4_g = tf.nn.conv2d(relu3_g, conv4w, [1, 1, 1, 1], padding='VALID')
            bias4_g = tf.nn.bias_add(conv4_g, conv4b)
            relu4_g = tf.nn.relu(bias4_g)

            conv1_l = tf.nn.conv2d(rgb_l, conv1w, [1, 2, 2, 1], padding='VALID')
            bias1_l = tf.nn.bias_add(conv1_l, conv1b)
            relu1_l = tf.nn.relu(bias1_l)
            lrn1_l = tf.nn.local_response_normalization(relu1_l, depth_radius=raidus, alpha=alpha, beta=beta,
                                                        bias=bias)
            pool1_l = self.max_pool(lrn1_l, 'pool1_l')

            conv2_l = tf.nn.conv2d(pool1_l, conv2w, [1, 2, 2, 1], padding='VALID')
            bias2_l = tf.nn.bias_add(conv2_l, conv2b)
            relu2_l = tf.nn.relu(bias2_l)
            lrn2_l = tf.nn.local_response_normalization(relu2_l, depth_radius=raidus, alpha=alpha, beta=beta,
                                                        bias=bias)
            pool2_l = self.max_pool(lrn2_l, 'pool2_l')

            conv3_l = tf.nn.conv2d(pool2_l, conv3w, [1, 1, 1, 1], padding='VALID')
            bias3_l = tf.nn.bias_add(conv3_l, conv3b)
            relu3_l = tf.nn.relu(bias3_l)

            conv4_l = tf.nn.conv2d(relu3_l, conv4w, [1, 1, 1, 1], padding='VALID')
            bias4_l = tf.nn.bias_add(conv4_l, conv4b)
            relu4_l = tf.nn.relu(bias4_l)

            fc_input1 = tf.reshape(relu4_g, [-1, 512])
            fc_input2 = tf.reshape(relu4_l, [-1, 512])

            fc_input = tf.concat([fc_input1, fc_input2], 1)

            fc1 = self.fc_layer(fc_input, 1024, 512, 'fc1')
            relu4 = tf.nn.relu(fc1, 'relu4')
            relu4 = tf.concat([relu4, a], 1)
            out = self.fc_layer(relu4, 515, 1, 'fc3')
            self.data_dict = None
            return out


    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)


    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def fc_layer_2(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var_2(in_size, out_size, name)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_fc_var_2(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.1, seed=1)
        # tf.truncated_norma
        weights = self.get_var(initial_value, name, 0, name+'_weights')
        initial_value = tf.truncated_normal([out_size], .0, .1)
        biases = self.get_var(initial_value, name, 1, name+'_biases')
        return weights, biases


    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.01, seed=1)
        # tf.truncated_norma
        weights = self.get_var(initial_value, name, 0, name+'_weights')
        initial_value = tf.truncated_normal([out_size], .0, .01)
        biases = self.get_var(initial_value, name, 1, name+'_biases')
        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = self.sess.run(initial_value)
        var = tf.Variable(value, name=var_name)
        return var

    def save_npy(self, actor_path='model/actor.npy', critic_path='critic/critic.npy'):
        critic_dict = {}
        i = 0
        critic_model = self.sess.run(self.ce_params)
        for name in critic_list:
            critic_dict[name] = {}
            for idx in range(2):
                critic_dict[name][idx] = critic_model[i]
                i += 1

        actor_dict = {}
        i = 0
        actor_model = self.sess.run(self.ae_params)
        for name in actor_list:
            actor_dict[name] = {}
            for idx in range(2):
                actor_dict[name][idx] = actor_model[i]
                i += 1

        np.save(actor_path, actor_dict)
        np.save(critic_path, critic_dict)
        return actor_path, critic_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x*y, v.get_shape().as_list())
        return count


def main():
    # avoid printing TF debugging information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    args = paras_init()
    var = 0.5

    train_ilsvrc_data_path = 'ilsvrc_train.json'
    ilsvrc_home = 'VID'
    train_dataset = ILSVRCDataset(train_ilsvrc_data_path, ilsvrc_home + '/train')

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = args.train_conserveMemory
    with tf.Session(config=tf_config) as sess:

        ddpg = DDPG(sess, './model/vggm1-4.npy', './model/vggm1-4.npy')

        tf.summary.scalar('loss', ddpg.td_error)
        merged_summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter("logs/", sess.graph)

        sess.run(tf.global_variables_initializer())
        reward_100 = 0
        ddpg.take_up()
        for train_step in range(1, 250000):
            frame_name_list, gt, length = train_dataset.next()

            img = skimage.io.imread(frame_name_list[0])
            img_size = img.shape
            if img_size.__len__() == 2:
                img = np.stack([img, img, img], 2)

            ground_th = gt[0]
            rate = ground_th[2] / ground_th[3]

            pos = ground_th
            reward_all = 0

            for init_num in range(1):
                actor_samples = np.round(
                    genSample('gaussian', ground_th, img_size, 64, [0.6, 1], 0.2, 1.1))
                distance = cal_distance(actor_samples, np.tile(pos, [actor_samples.shape[0], 1]))
                distance[distance > 1] = 1
                distance[distance < -1] = -1
                batch_g, batch_l = getbatch(img, actor_samples, args)
                ddpg.init_actor(batch_g, batch_l, distance)


            for frame in range(1, length):
                img = skimage.io.imread(frame_name_list[frame])
                if img_size.__len__() == 2:
                    img = np.stack([img, img, img], 2)

                pos_ = pos

                img_crop_l, img_crop_g = crop_image(img, pos)
                imo_l = np.array(img_crop_l).reshape(1, 107, 107, 3)
                imo_g = np.array(img_crop_g).reshape(1, 107, 107, 3)

                deta_pos = sess.run(ddpg.a, feed_dict={ddpg.Sl: imo_l, ddpg.Sg: imo_g})[0]

                if np.random.random(1) < var:
                    deta_pos_ = cal_distance(np.vstack([pos, pos]), np.vstack([gt[frame], gt[frame]]))
                    if np.max(abs(deta_pos_)) < 1:
                        deta_pos = deta_pos_[0]

                if deta_pos[2] > 0.05 or deta_pos[2] < -0.05:
                    deta_pos[2] = 0

                pos_ = move_crop(pos_, deta_pos, img_size, rate)

                img_crop_l_, img_crop_g_ = crop_image(img, pos_)
                imo_l_ = np.array(img_crop_l_).reshape(1, 107, 107, 3)
                imo_g_ = np.array(img_crop_g_).reshape(1, 107, 107, 3)
                r = _compute_iou(pos_, gt[frame])

                if r > 0.7:
                    reward = 1
                else:
                    reward = -1

                ddpg.store_transition(imo_g, imo_l, deta_pos, reward, imo_g_, imo_l_)
                reward_all += reward
                pos = pos_

            ddpg.learn()

            reward_100 += reward_all

            rs = sess.run(merged_summary_op, {ddpg.Sg: imo_g, ddpg.Sl: imo_l, ddpg.a: np.array(deta_pos).reshape([1,3]), ddpg.R: np.array(reward).reshape([1,1]), ddpg.Sg_: imo_g_, ddpg.Sl_: imo_l_})
            summary_writer.add_summary(rs, train_step)

            if train_step % 100 == 0:
                print train_step, reward_100, 'td_error', ddpg.show_critic_loss()
                reward_100 = 0

            if train_step % 10000 == 0:
                ddpg.save_npy('./model/Actor' + '%d' % train_step + '.npy',
                              './model/Critic' + '%d' % train_step + '.npy')

            if train_step % 10000 == 0:
                var = var * 0.95


def genSample(type, ground_th, img_size, sample_num, sample_range,
              trans_f, scale_f):
    h = img_size[0]
    w = img_size[1]

    bb = np.array([ground_th[0]+ ground_th[2]/2., ground_th[1]+ground_th[3]/2., ground_th[2], ground_th[3]])
    samples = np.tile(bb, [sample_num, 1])
    if type is 'gaussian':
        samples[:, 0:2] = \
            samples[:,0:2] + trans_f * np.round(np.mean(ground_th[2:3]))* np.maximum(-1,np.minimum(1, 0.5 * np.random.randn(sample_num,2)))
        samples[:, 2:] = \
            samples[:,2:] * np.tile(1.05**(scale_f*np.maximum(-1,np.minimum(1, 0.5 * np.random.randn(sample_num,1)))),[1,2])
    #if type is 'uniform':
    elif type is 'uniform':
        samples[:, 0:2] = \
            samples[:,0:2] + trans_f*np.round(np.mean(ground_th[2:3]))* (np.random.randn(sample_num,2)*2-1)
        samples[:, 2:] = \
            samples[:,2:] * np.tile(1.05**(scale_f*np.random.randn(sample_num,1)*2-1),[1,2])
    elif type is 'whole':
        range = np.round([ground_th[2]/2, ground_th[3]/2, w-ground_th[2]/2, h-ground_th[3]/2])
        stride = np.round([ground_th[2]/5, ground_th[3]/5])
        dx,dy,ds = np.meshgrid(np.arange(range[0],range[2],stride[0]),np.arange(range[1],range[3],stride[1]), np.arange(-5,5))
        windows = np.transpose(np.vstack(((dx.flatten()), (dy.flatten()), ground_th[2]*(1.05**(ds.flatten())), ground_th[3]*(1.05**(ds.flatten())))))
        samples = windows[np.random.choice(windows.shape[0], sample_num),:]

    samples[:, 2] = np.maximum(10, np.minimum(w - 10, samples[:, 2]))
    samples[:, 3] = np.maximum(10, np.minimum(h - 10, samples[:, 3]))

    bb_samples = np.transpose(np.vstack((np.transpose(samples[:,0]-samples[:,2]/2), np.transpose(samples[:,1]-samples[:,3]/2), np.transpose(samples[:,2]), np.transpose(samples[:,3]))))
    bb_samples[:, 0] = np.maximum(1 - bb_samples[:, 2] / 2., np.minimum(w - bb_samples[:, 2] / 2., bb_samples[:, 0]))
    bb_samples[:, 1] = np.maximum(1 - bb_samples[:, 3] / 2., np.minimum(h - bb_samples[:, 3] / 2., bb_samples[:, 1]))
    samples = np.round(bb_samples)

    r = func_iou(samples, ground_th)
    if type is 'gaussian':
        idx_temp = np.where(r>sample_range[0])
        idx = idx_temp[0]
        samples = samples[idx][:]
    else:
        idx_temp = np.where(r < sample_range[1])
        idx = idx_temp[0]
        samples = samples[idx][:]

    return samples

def func_iou(bb, gtb):
    gtbb = np.tile(gtb,[bb.shape[0],1])
    iou = np.zeros((bb.shape[0],1))
    for i in range(bb.shape[0]):
        iw = min(bb[i][2]+bb[i][0],gtbb[i][2]+gtbb[i][0]) - max(bb[i][0],gtbb[i][0]) + 1
        ih = min(bb[i][3]+bb[i][1],gtbb[i][3]+gtbb[i][1]) - max(bb[i][1],gtbb[i][1]) + 1
        if iw>0 and ih>0:
            ua = (bb[i][2]+1)*(bb[i][3]+1) + (gtbb[i][2]+1)*(gtbb[i][3]+1) - iw*ih
            iou[i][:] = iw*ih/ua
    return iou

def getbatch(img, boxes, args):
    crop_size = args.sampling_input_size

    num_boxes = boxes.shape[0]
    imo_g = np.zeros([num_boxes, crop_size, crop_size, 3])
    imo_l = np.zeros([num_boxes, crop_size, crop_size, 3])

    for i in range(num_boxes):

        bbox = boxes[i]
        img_crop_l, img_crop_g = crop_image(img, bbox)

        imo_g[i] = img_crop_g
        imo_l[i] = img_crop_l

    return imo_g, imo_l

def _compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou

def move_crop(pos_, deta_pos, img_size, rate):
    flag = 0
    if deta_pos[2] > 0.05 or deta_pos[2] < -0.05:
        deta_pos[2] = 0

    if pos_.shape.__len__() == 1:
        pos_ = np.array(pos_).reshape([1,4])
        deta_pos = np.array(deta_pos).reshape([1, 3])
        flag = 1
    pos_deta = deta_pos[:, 0:2] * pos_[:, 2:]
    pos = np.copy(pos_)
    center = pos[:, 0:2] + pos[:, 2:4] / 2
    center_ = center - pos_deta
    pos[:, 2] = pos[:, 2] * (1 + deta_pos[:, 2])
    pos[:, 3] = pos[:, 3] * (1 + deta_pos[:, 2])



    pos[pos[:, 2] < 10, 2] = 10
    pos[pos[:, 3] < 10, 3] = 10


    pos[:, 0:2] = center_ - pos[:, 2:4] / 2

    pos[pos[:, 0] + pos[:, 2] > img_size[1], 0] = \
        img_size[1] - pos[pos[:, 0] + pos[:, 2] > img_size[1], 2] - 1
    pos[pos[:, 1] + pos[:, 3] > img_size[0], 1] = \
        img_size[0] - pos[pos[:, 1] + pos[:, 3] > img_size[0], 3] - 1
    pos[pos[:, 0] < 0, 0] = 0
    pos[pos[:, 1] < 0, 1] = 0

    pos[pos[:, 2] > img_size[1], 2] = img_size[1]
    pos[pos[:, 3] > img_size[0], 3] = img_size[0]
    if flag == 1:
        pos = pos[0]

    return pos

def cal_distance(samples, ground_th):
    distance = samples[:, 0:2] + samples[:, 2:4] / 2 - ground_th[:, 0:2] - ground_th[:, 2:4] / 2
    distance = distance / samples[:, 2:4]
    rate = ground_th[:, 3] / samples[:, 3]
    rate = np.array(rate).reshape(rate.shape[0], 1)
    rate = rate - 1.0
    distance = np.hstack([distance, rate])
    return distance

def crop_image(img, bbox, img_size=107, padding=0, valid=False):
    x, y, w, h = np.array(bbox, dtype='float32')

    half_w, half_h = w / 2, h / 2
    center_x, center_y = x + half_w, y + half_h

    if padding > 0:
        pad_w = padding * w / img_size
        pad_h = padding * h / img_size
        half_w += pad_w
        half_h += pad_h

    img_h, img_w, _ = img.shape
    min_x = int(center_x - half_w + 0.5)
    min_y = int(center_y - half_h + 0.5)
    max_x = int(center_x + half_w + 0.5)
    max_y = int(center_y + half_h + 0.5)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')
        # try:
        cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]
        # except:
        #     a= 1

    scaled_l = imresize(cropped, (img_size, img_size))

    min_x = int(center_x - w + 0.5)
    min_y = int(center_y - h + 0.5)
    max_x = int(center_x + w + 0.5)
    max_y = int(center_y + h + 0.5)

    if min_x >= 0 and min_y >= 0 and max_x <= img_w and max_y <= img_h:
        cropped = img[min_y:max_y, min_x:max_x, :]

    else:
        min_x_val = max(0, min_x)
        min_y_val = max(0, min_y)
        max_x_val = min(img_w, max_x)
        max_y_val = min(img_h, max_y)

        cropped = 128 * np.ones((max_y - min_y, max_x - min_x, 3), dtype='uint8')
        cropped[min_y_val - min_y:max_y_val - min_y, min_x_val - min_x:max_x_val - min_x, :] \
            = img[min_y_val:max_y_val, min_x_val:max_x_val, :]

    scaled_g = imresize(cropped, (img_size, img_size))

    return scaled_l, scaled_g

if __name__ == '__main__':
    sys.exit(main())
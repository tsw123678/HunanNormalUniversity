from __future__ import division
#import tensorflow as tf
import numpy as np
import time
from data_gen.UAVModel import UAV
# from UAVModel import UAV
import os
import shutil
import torch
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
np.random.seed(1)
#tf.set_random_seed(1)
#tf.random.set_seed(1)
tf.random.set_random_seed(1)
MAX_EPS = 10
MAX_STEP = 50
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.01      # soft replacement
memory_capa = 200000
BATCH_SIZE = 256
s_dim = 3
a_dim = 2
a_bound = 1
###############################
kp = 100
kq = 1
M = 100

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

# irs = [[100,0,50], [300,0,50],[500,0,50],[100,200,50],[300,200,50],[500,200,50]]
# ue = [[100,50],[300,50],[500,50],[100,150],[300,150],[500,150]]
irs = [[2,0,10], [6,0,10],[10,0,10],[2,4,10],[6,4,10],[10,4,10]]
ue = [[2,1],[6,1],[10,1],[2,3],[6,3],[10,3]]
# irs = [[2,0,5], [6,0,5],[10,0,5],[2,4,5],[6,4,5],[10,4,5]]
# ue = [[2,1],[6,1],[10,1],[2,3],[6,3],[10,3]]
L = len(irs)
N = len(ue)
save_name = 'DDPG_' + str(kp) + '_' + str(kq) + '_' + str(M) + '_' + str(L)
battery_max = 2*1e4
H = 20
T = 200
d_max = 30  #水平最大位移
P_max = 0.01# 0.1 watt
noise = 10**(-11)# -80 dbm
#pl = 0.01# -20db
pl = 0.1# -10db
beta = np.power(10, 0.3)
rho = 1# wavelength
d_as = rho/2# distance antenna separation
imag = 1j
uav = UAV()

ch_ur_tf = tf.placeholder(shape=[L * M, 1], dtype=tf.complex64)
ch_re_matrix_tf = tf.placeholder(shape=[L * M, 1], dtype=tf.complex64)
ph_matrix_tf = tf.placeholder(shape=[L * M, L * M], dtype=tf.complex64)
z = tf.abs(tf.linalg.det(tf.transpose(ch_re_matrix_tf, conjugate=True) @ ph_matrix_tf @ ch_ur_tf))
#z = tf.transpose(ch_re_matrix_tf, conjugate=True) @ ph_matrix_tf @ ch_ur_tf


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((memory_capa, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        action = self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
        return action#self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(memory_capa, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % memory_capa  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 300, activation=tf.nn.relu, name='l1', trainable=trainable)
            net1 = tf.layers.dense(net, 300, activation=tf.nn.relu, name='l2', trainable=trainable)
            a = tf.layers.dense(net1, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            #tf.multiply(a, self.a_bound, name='scaled_a')
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 300
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net1 = tf.layers.dense(net, 300, activation=tf.nn.relu, name='c2', trainable=trainable)
            return tf.layers.dense(net1, 1, trainable=trainable)  # Q(s,a)

    def save(self, name):
        #name = 'dqn'
        path = './variable_' + name
        saver = tf.train.Saver()
        if os.path.isdir(path): shutil.rmtree(path)
        os.mkdir(path)
        ckpt_path = os.path.join(path, name + '.ckpt')
        save_path = saver.save(self.sess, ckpt_path, write_meta_graph=True)
        print("\nSave Model %s\n" % save_path)

    def restore(self, name):
        path = './variable_' + name
        saver = tf.train.Saver()
        #saver = tf.train.Saver(save_relative_paths=True)
        saver.restore(self.sess, tf.train.latest_checkpoint(path))
        #saver.restore(sess, tf.train.latest_checkpoint(path))
        print('success')



ddpg = DDPG(a_dim, s_dim, a_bound)


def cal_dis_UR(uav, irs):
    L = len(irs)
    dis = np.zeros(L)
    for l in range(L):
        dis[l] = np.sqrt(np.square(uav.x-irs[l][0])+np.square(uav.y-irs[l][1])+np.square(uav.z-irs[l][2]))
    return dis

def cal_dis_RE(irs, ue):
    L = len(irs)
    dis = np.zeros((L, N))
    for l in range(L):
        for n in range(N):
            dis[l][n] = np.sqrt(np.square(irs[l][0]-ue[n][0])+np.square(irs[l][1]-ue[n][1])+np.square(irs[l][2]))
    return dis

def cal_ch_UR(dis_ur, uav, irs):
    L = len(irs)
    ch = np.zeros(L*M, dtype=complex)
    for l in range(L):
        for m in range(M):
            k1 = np.sqrt(pl/np.square(dis_ur[l]))
            k2 = (irs[l][0]-uav.x)/dis_ur[l]
            #angle = abs(2*np.pi/rho*d_as*m*k2)
            angle = 2 * np.pi / rho * d_as * m * k2
            ch[l*M+m] = k1 * np.power(np.e, -imag * angle)
    return ch

def cal_ch_RE(dis_re, irs):
    L = len(irs)
    ch = np.zeros((N, L*M), dtype=complex)
    for n in range(N):
        for l in range(L):
            for m in range(M):
                k1 = np.sqrt(pl * np.power(dis_re[l][n], -2.8))
                k2 = (ue[n][0]-irs[l][0]) / dis_re[l][n]
                #angle = abs(2 * np.pi / rho * d_as * m *k2)
                angle = 2 * np.pi / rho * d_as * m * k2
                ch[n][l * M + m] = k1 * np.power(np.e, -imag*angle)
    return ch


def cal_PH(uav, irs, ue, dis_ur, dis_re):
    L = len(irs)
    ph = np.zeros((N, L*M, L*M), dtype=complex)
    for n in range(N):
        for l in range(L*M):
            k_ur = -2 * np.pi / rho * d_as * (l % M) * (irs[l // M][0] - uav.x) / dis_ur[l // M]
            k_re = -2 * np.pi / rho * d_as * (l % M) * (ue[n][0] - irs[l // M][0]) / dis_re[l // M][n]
            if k_ur<0:
                while True:
                    k_ur+=2*np.pi
                    if k_ur>=0:
                        break
            elif k_ur>=2*np.pi:
                while True:
                    k_ur-=2*np.pi
                    if k_ur<2*np.pi:
                        break
            if k_re<0:
                while True:
                    k_re+=2*np.pi
                    if k_re>=0:
                        break
            elif k_re>=2*np.pi:
                while True:
                    k_re-=2*np.pi
                    if k_re<2*np.pi:
                        break
            ttt = k_re - k_ur
            if ttt<0:
                ttt+=2*np.pi
            elif ttt>2*np.pi:
                ttt-=2*np.pi
            ph[n][l][l] = np.power(np.e, imag * (ttt))
    return ph



'''def cal_data_rate(ch_ur, ch_re, phase_shift):
    rate = np.zeros(N)
    L = len(irs)
    ch_ur = np.reshape(ch_ur, (L*M, 1))
    ch_ur = np.asmatrix(ch_ur)
    for n in range(N):
        ph_matrix = np.asmatrix(phase_shift[n])
        ch_re_matrix = np.asmatrix(np.reshape(ch_re[n], (L*M, 1)))
        #snr = np.transpose(ch_re_matrix) * ph_matrix * ch_ur
        snr = ch_re_matrix.H * ph_matrix * ch_ur
        snr = np.abs(np.linalg.det(snr))
        #rate_val_b = np.log2(1+P_max*np.square(snr)/(noise))
        rate_val_b = np.log2(1 + P_max * snr/noise)
        rate[n] = rate_val_b
    return rate'''

def cal_data_rate(ch_ur, ch_re, phase_shift):
    SNR = np.zeros(N)
    rate = np.zeros(N)
    L = len(irs)
    ch_ur = np.reshape(ch_ur, (L*M, 1))
    ch_ur = np.asmatrix(ch_ur)
    for n in range(N):
        snr = sess.run(z, feed_dict={ch_re_matrix_tf: np.reshape(ch_re[n], (L*M, 1)), ch_ur_tf: np.reshape(ch_ur, (L*M, 1)), ph_matrix_tf: phase_shift[n]})
        SNR[n] = snr
        rate_val_b = np.log2(1 + P_max * snr / noise)
        rate[n] = rate_val_b
    return rate, SNR

def cal_fairness(serve_count_eps, selected_ue):
    serve_count_eps[selected_ue] += 1
    son, mom = 0, 0
    for n in range(N):
        son += serve_count_eps[n]
        mom += np.square(serve_count_eps[n])
    fair = np.square(son)/(N*mom)
    return fair, serve_count_eps

def cal_h(channel_num,num_tap):
    assert num_tap == 3
    data = dataset()
    channel_list = []
    for i in range(channel_num):  # default=100/20
        channel_list_total = torch.randperm(len(data))  # sampling with replacement 重复抽样  Returns a random permutation of integers from 0 to n - 1.
        current_channel_ind = channel_list_total[i]
        current_channel = data[current_channel_ind]
        channel_list.append(current_channel)
    return channel_list

def dataset():
    count = 0
    var = 0.05
    start_flag = False
    channel_list = []
    rate_all = []
    fair_all = []
    reward_all = []
    loss_all = []
    reach_out = 0
    ddpg.save(save_name)
    ddpg.restore(save_name)
    for eps in range(MAX_EPS):
        uav.reset(10, 10, H, battery_max, d_max, out=False)
        step = 0
        rew_eps = 0
        fair_eps = 0
        ene_eps = 0
        reward_eps = 0
        rate_eps = 0
        rate_eps_ran = 0
        serve_count_eps = np.zeros(N)
        out_count = 0
        sel_all = []
        for step in range(MAX_STEP):
        #while uav.battery >= 0:
            state = np.array([uav.x / 600, uav.y / 200, uav.battery / battery_max])
            action = ddpg.choose_action(s=state)
            action = np.clip(np.random.normal(action, var), -1, 1)
            con_ene = uav.move(action=action)
            dis_ur = cal_dis_UR(uav, irs)
            dis_re = cal_dis_RE(irs, ue)
            ch_ur = cal_ch_UR(dis_ur, uav, irs)
            ch_re = cal_ch_RE(dis_re, irs)
            phase_shift = cal_PH(uav, irs, ue, dis_ur, dis_re)
            data_rate, SNR = cal_data_rate(ch_ur, ch_re, phase_shift)
            h = torch.tensor(SNR)
            channel_list.append(h)
            selected_ue = list(data_rate).index(max(data_rate))
            fair, serve_count_eps = cal_fairness(serve_count_eps, selected_ue)
            sel_all.append(fair)
            fair_eps += fair
            ene_eps += con_ene
            rate_eps += data_rate[selected_ue]
            if uav.out is True:
                reward = (kp * fair + kq * data_rate[selected_ue])-100
                out_count += 1
            else:
                reward = (kp * fair + kq * data_rate[selected_ue])
            reward_eps += reward
            rew_eps += reward
            state_ = np.array([uav.x / 600, uav.y / 200, uav.battery / battery_max])
            count += 1
            step += 1
        #print('SNR:%s, phase_shift:%s, ch_ur:%s, ch_re:%s, eps:%s, percent:%s, step:%s, reach:%s, serve:%s, uav:%s, fair:%s, rate:%s, energy:%s, reward:%s, out:%s, start_learning:%s' % (SNR, phase_shift, ch_ur, ch_re, eps, count / memory_capa, step, reach_out, serve_count_eps, [uav.x, uav.y], fair_eps / step, rate_eps / step, ene_eps / step, reward_eps / step, out_count/step, start_flag))
        rate_all.append(rate_eps / step)
        fair_all.append(fair_eps / step)
        reward_all.append(reward_eps / step)
    #print(channel_list, len(channel_list))
    return  channel_list

# if __name__ == '__main__':
#     import platform
#
#     print('系统:', platform.system())
#     time_start = time.clock()
#     channel_list = dataset()
#     print(channel_list)
#     print('\n')
#     channel_list1 = cal_h(20,3)
#     print(channel_list1,len(channel_list1))
#     time_end = time.clock()
#     print('time cost', time_end - time_start, 's')







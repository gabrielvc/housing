import tensorflow as tf
import numpy as np


class NN_mlp:
    def __init__(self, size_layers, actfun, cost):
        self.weights = np.array([np.random.normal(size=(size_layers[i], size_layers[i+1])) for i in range(len(size_layers)-1)])
        self.bias = np.array([0.1*np.ones(size_layers[i]) for i in range(1, len(size_layers))])
        self.actfun = actfun
        self.cost = cost

    def train(self, X, Y, n_iter=100, min_step=1e-2, sum_flag=False, sum_dir='', cross_flag=False, X_cross=[], Y_cross=[], drop_layers=[], prob_drops=[]):
        x = tf.placeholder(tf.float64, [None, X.shape[1]])
        y_ = tf.placeholder(tf.float64, [None, Y.shape[1]])

        w_layers = [tf.Variable(self.weights[i]) for i in range(len(self.weights))]
        b_layers = [tf.Variable(self.bias[i]) for i in range(len(self.bias))]
        activations = np.array([x])
        keep_prob = tf.placeholder(tf.float64, shape=None)
        if 0 in drop_layers:
            activations = np.array([tf.nn.dropout(x,keep_prob[0])])

        for i in range(len(self.weights)-1):
            activate = getattr(tf.nn, self.actfun[i])(tf.matmul(activations[-1], w_layers[i])+b_layers[i])
            if (i+1) in drop_layers:
                ind = np.where(drop_layers ==(i+1))[0][0]
                activate = tf.nn.dropout(activate, keep_prob[ind])

            activations = np.append(activations, [activate], axis=0)

        y = tf.matmul(activations[-1], w_layers[-1])+b_layers[-1]

        if self.cost == 'MSE':
            cost_fun = tf.squeeze(tf.reduce_mean(tf.square(y - y_)))
        if self.cost == 'MAE':
            cost_fun = tf.squeeze(tf.reduce_mean(tf.abs(y - y_)))
        if self.cost == 'cross_entropy_softmax':
            cost_fun = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

        train_step = tf.train.AdamOptimizer(min_step).minimize(cost_fun)

        W_nn = w_layers
        b_nn = b_layers

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        ssum = tf.summary.scalar('cost_function', cost_fun)
        #cross_sum = tf.scalar_summary('Validation error',cost_fun)
        train_writer = tf.summary.FileWriter(sum_dir+'/train', sess.graph)
        cross_writer = tf.summary.FileWriter(sum_dir+'/validation', sess.graph)


        for i in range(n_iter):
            sess.run(train_step, feed_dict={x: X, y_: Y, keep_prob: prob_drops})

            if i % 100 and sum_flag:
                summary_train = sess.run(ssum, feed_dict={x: X, y_: Y, keep_prob: prob_drops})
                if cross_flag:
                    summary_cross = sess.run(ssum, feed_dict={x: X_cross, y_: Y_cross, keep_prob: np.ones(len(prob_drops))})
                    cross_writer.add_summary(summary_cross, i)
                train_writer.add_summary(summary_train, i)
        self.weights = np.array(sess.run(W_nn))
        self.bias = np.array(sess.run(b_nn))

    def predict(self, X):
        a = tf.placeholder(tf.float64, shape=[None, X.shape[1]])
        activations = np.array([a])
        for i in range(len(self.weights)-1):
            activations = np.append(activations, [getattr(tf.nn, self.actfun[i])(tf.matmul(activations[-1], self.weights[i])+self.bias[i])], axis=0)
        y_ = tf.matmul(activations[-1], self.weights[-1])+self.bias[-1]
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        y = sess.run(y_, feed_dict={a: X})
        return np.array([i[0] for i in y])

# -*- coding: utf_8 -*-
import sys
import numpy as np
from sklearn.base import BaseEstimator

from chainer import Variable
import chainer.functions as F
from chainer import cuda


class NNmanager (BaseEstimator):
    def __init__(self, model, optimizer, lossFunction, **params):
        # 学習器の初期化
        # ネットワークの定義
        self.model = model
        # オプティマイザの設定
        self.optimizer = optimizer
        self.optimizer.setup(self.model)
        # 損失関数の設定
        self.lossFunction = lossFunction
        # epochの設定
        self.epoch = params['epoch'] if 'epoch' in params else 20
        # バッチサイズの設定
        self.batchsize = params['batchsize'] if 'batchsize' in params else 100
        # ロギングの設定
        self.logging = params['logging'] if 'logging' in params else False

    def fit(self, x_train, y_train):
        self.runEpoch(x_train, y_train)
        return self

    def predict(self, x_test):
        output = self.forward(x_test, train=False)
        return self.trimOutput(output)

    def trimOutput(self, output):
        # 結果を整形したいときなど。二値分類のときにグニャらせたりとか
        return output.data

    # 順伝播・逆伝播。ネットワーク構造に応じて自分で定義しないとエラーを吐くようにしておく
    def forward(self, x_data, train):
        raise NotImplementedError("`forward` method is not implemented.")
        # x = Variable(x_data)
        # h1 = F.relu(self.model.l1(x))
        # h2 = F.relu(self.model.l2(h1))
        # y_predict = self.model.l3(h2)
        # return y_predict

    def backward(self, y_predict, y_data):
        y = Variable(y_data)
        loss = self.lossFunction(y_predict, y)
        accuracy = 1000.0 / self.lossFunction(y_predict, y) #F.accuracy(y_predict, y)
        loss.backward()
        return loss, accuracy

    def setLogger(self, logging):
        self.logging = logging

    def runEpoch(self, x_train, y_train):
        best_loss = sys.maxint
        last_update = 0
        epoch = 0
        while True:
            epoch+=1
            mean_loss, mean_accuracy = self.epochProcess(x_train, y_train)
            if best_loss > mean_loss:
                best_loss = mean_loss
                last_update = epoch
            if(self.logging):
                logFormat = "epoch\t%d\tmean_loss\t%f\tmean_accuracy\t%f\tbest_loss\t%f"
                print logFormat % (epoch, mean_loss, mean_accuracy, best_loss)
            if epoch - last_update > self.epoch:
                return

    def epochProcess(self, x_train, y_train):
        trainsize = len(y_train)
        indexes = np.random.permutation(trainsize)
        sum_loss = 0
        sum_accuracy = 0

        for i in xrange(0, trainsize, self.batchsize):
            if isinstance(x_train, cuda.cupy.ndarray):
                x_batch = cuda.cupy.array(x_train.get()[indexes[i: i + self.batchsize]])
                y_batch = cuda.cupy.array(y_train.get()[indexes[i: i + self.batchsize]])
            else:
                x_batch = x_train[indexes[i: i + self.batchsize]]
                y_batch = y_train[indexes[i: i + self.batchsize]]

            self.optimizer.zero_grads()
            y_predict = self.forward(x_batch, train=True)
            loss, accuracy = self.backward(y_predict, y_batch)
            self.optimizer.update()
            sum_loss += loss.data * self.batchsize
            sum_accuracy += accuracy.data * self.batchsize

        mean_loss = sum_loss / trainsize
        mean_accuracy = sum_accuracy / trainsize
        return mean_loss, mean_accuracy

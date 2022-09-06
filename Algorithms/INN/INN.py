from IPython.display import HTML
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import itertools
from flow import *
from utils import *
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.keras.optimizers import Adam,RMSprop
import pandas as pd
import tensorflow as tf

class Trainer(tfk.Model):
    def __init__(self, model, x_dim, y_dim, z_dim, tot_dim, 
                 n_couple_layer, n_hid_layer, n_hid_dim, shuffle_type='reverse'):
        super(Trainer, self).__init__()
        self.model = model
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.tot_dim = tot_dim
        self.x_pad_dim = tot_dim - x_dim
        self.y_pad_dim = tot_dim - (y_dim + z_dim)
        self.n_couple_layer = n_couple_layer
        self.n_hid_layer = n_hid_layer
        self.n_hid_dim = n_hid_dim
        self.shuffle_type = shuffle_type

        self.w1 = 5.
        self.w2 = 1.
        self.w3 = 10.
        self.loss_factor = 1.
        self.loss_fit = MSE
        self.loss_latent = MMD_multiscale
        self.loss_backward = MSE

    def train_step(self, data):
        x_data, y_data = data
        x = x_data[:, :self.x_dim]
        y = y_data[:, -self.y_dim:]
        z = y_data[:, :self.z_dim]
        y_short = tf.concat([z, y], axis=-1)

        # Forward loss
        with tf.GradientTape() as tape:
            y_out = self.model(x_data)    
            pred_loss = self.w1 * self.loss_fit(y_data[:,self.z_dim:], y_out[:,self.z_dim:]) # [zeros, y] <=> [zeros, yhat]
            output_block_grad = tf.concat([y_out[:,:self.z_dim], y_out[:, -self.y_dim:]], axis=-1) # take out [z, y] only (not zeros)
            latent_loss = self.w2 * self.loss_latent(y_short, output_block_grad) # [z, y] <=> [zhat, yhat]
            forward_loss = pred_loss + latent_loss
        grads_forward = tape.gradient(forward_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads_forward, self.model.trainable_weights))

        # Backward loss
        with tf.GradientTape() as tape:
            x_rev = self.model.inverse(y_data)
            #rev_loss = self.w3 * self.loss_factor * self.loss_fit(x_rev, x_data)
            rev_loss = self.w3 * self.loss_factor * self.loss_backward(x_rev, x_data)
        grads_backward = tape.gradient(rev_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads_backward, self.model.trainable_weights)) 

        total_loss = forward_loss + latent_loss + rev_loss
        return {'total_loss': total_loss,
                'forward_loss': forward_loss,
                'latent_loss': latent_loss,
                'rev_loss': rev_loss}

    def test_step(self, data):
        x_data, y_data = data
        return NotImplementedError


def prepare_dataset(X,Y,n_batch,pad_dim,z_dim):

    pad_x = np.zeros((X.shape[0], pad_dim))
    x_data = np.concatenate([X, pad_x], axis=-1).astype('float32')
    print(x_data.shape)
    z = np.random.multivariate_normal([0.]*z_dim, np.eye(z_dim), X.shape[0])
    y_data = np.concatenate([z, Y], axis=-1).astype('float32')
    print(y_data.shape)
    # Make dataset generator
    x_data = tf.data.Dataset.from_tensor_slices(x_data)
    y_data = tf.data.Dataset.from_tensor_slices(y_data)
    dataset = (tf.data.Dataset.zip((x_data, y_data))
            .shuffle(buffer_size=X.shape[0])
            .batch(n_batch, drop_remainder=True)
            .repeat())
    
    return dataset

def INN(X,Y,n_couple_layer = 4,n_hid_layer = 2,n_hid_dim = 32,n_batch = 128,n_epoch = 100,n_display = 5):

    x_dim = X.shape[1]
    y_dim = Y.shape[1]
    z_dim = 5
    n_data = X.shape[0]
    tot_dim = y_dim + z_dim
    pad_dim = tot_dim - x_dim
    dataset = prepare_dataset(X,Y,n_batch,pad_dim,z_dim)
    trainer = Trainer(model, x_dim, y_dim, z_dim, tot_dim, n_couple_layer, n_hid_layer, n_hid_dim)
    trainer.compile(optimizer='RMSprop')
    hist = trainer.fit(dataset,
                   batch_size=n_batch,
                   epochs=n_epoch,
                   steps_per_epoch=n_data//n_batch, 
                   verbose=0)

    f = open("Results/Training/INN.sav","wb")
    pickle.dump(model,trainer)
    f.close()

    
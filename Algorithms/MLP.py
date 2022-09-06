import tensorflow as tf
import levenberg_marquardt as lm
from keras.models import Sequential
from keras.layers import Dense,BatchNormalization
from keras import optimizers,regularizers,initializers
from keras.optimizers import SGD,Adam

def MLP_Model(X_train,Y_train):

    model = Sequential()
    model.add(Dense(100, input_dim=X_train.shape[1],
                    kernel_initializer=initializers.RandomUniform(minval=0, maxval=1, seed=None), 
                    activation='relu',
                    kernel_regularizer=regularizers.l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(Y_train.shape[1], activation='relu'))
    opt = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    model.summary()

    model_wrapper = lm.ModelWrapper(
        tf.keras.models.clone_model(model))

    model_wrapper.compile(
        optimizer=SGD(learning_rate=0.1),
        loss=lm.MeanSquaredError(),metrics=['accuracy'])

    return model_wrapper





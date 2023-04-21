import sys
from tensorflow.keras.layers import concatenate, Dot, Reshape
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from tqdm.keras import TqdmCallback
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import livelossplot
from tensorflow.keras.callbacks import ModelCheckpoint, History
from utils import *


# to fix error "'TqdmCallback' object has no attribute '_implements_train_batch_hooks'"
class TqdmCallbackFix(TqdmCallback):
    def _implements_train_batch_hooks(self): return True
    def _implements_test_batch_hooks(self): return True
    def _implements_predict_batch_hooks(self): return True

# build fusion-net for bacterial viability detection to optimize its hyperparameters.
def create_fusionnet_model(params):
    """This model creation function is used for hyperparameter optimization"""
    # Because Hyperas actually parse this method, all methods should be inside of this function
    # this code was changed from Rui Kang's scripts
    classes = 2
    nspecies = 6
    image_bands = 89
    image_size = 64
    filter_size = 64

    # setting up sub-models
    # 1D-CNN model
    spectral_cnn_model_in = Input((image_bands, 1))
    spectral_cnn_model = (Conv1D(128, 5, activation="relu"))(spectral_cnn_model_in)
    spectral_cnn_model = (Dropout(0.3))(spectral_cnn_model)
    spectral_cnn_model = (BatchNormalization())(spectral_cnn_model)
    spectral_cnn_model = (MaxPooling1D(pool_size=4))(spectral_cnn_model)
    spectral_cnn_model_flatten = (Flatten())(spectral_cnn_model)
    spectral_cnn_model_dense = (Dense(100, activation='relu'))(spectral_cnn_model_flatten )
    spectral_cnn_model_out = (Dense(classes, activation='softmax'))(spectral_cnn_model)

    spectral_cnn_model_final = Model(inputs=spectral_cnn_model_in, outputs=spectral_cnn_model_out)
    spectral_cnn_model_final.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

    # LSTM model - This is just a placeholder and is not used as weights for the network is zero
    lstm_shape_model_in = Input((1,9))

    # To build to ResNet50
    K.set_image_data_format('channels_last')
    input_shape = (image_size, image_size, 1)
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(filter_size, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=1)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[filter_size, filter_size, filter_size*2], stage=2, block='a', s=1)
    X = identity_block(X, 3, [filter_size, filter_size, filter_size*2], stage=2, block='b')
    X = identity_block(X, 3, [filter_size, filter_size, filter_size*2], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[filter_size*2, filter_size*2, filter_size*4], stage=3, block='a', s=2)
    X = identity_block(X, 3, [filter_size*2, filter_size*2, filter_size*4], stage=3, block='b')
    X = identity_block(X, 3, [filter_size*2, filter_size*2, filter_size*4], stage=3, block='c')
    X = identity_block(X, 3, [filter_size*2, filter_size*2, filter_size*4], stage=3, block='d')

    # Stage 4-5 were commented out based on results of hyperparameter optimization with this data
    # Stage 4
    # X = convolutional_block(X, f=3, filters=[filter_size*4, filter_size*4, filter_size*8], stage=4, block='a', s=2)
    # X = identity_block(X, 3, [filter_size*4, filter_size*4, filter_size*8], stage=4, block='b')
    # X = identity_block(X, 3, [filter_size*4, filter_size*4, filter_size*8], stage=4, block='c')
    # X = identity_block(X, 3, [filter_size*4, filter_size*4, filter_size*8], stage=4, block='d')
    # X = identity_block(X, 3, [filter_size*4, filter_size*4, filter_size*8], stage=4, block='e')
    # X = identity_block(X, 3, [filter_size*4, filter_size*4, filter_size*8], stage=4, block='f')
    # Stage 5
    # X = convolutional_block(X, f=3, filters=[filter_size*8, filter_size*8, filter_size*16], stage=5, block='a', s=2)
    # X = identity_block(X, 3, [filter_size*8, filter_size*8, filter_size*16], stage=5, block='b')
    # X = identity_block(X, 3, [filter_size*8, filter_size*8, filter_size*16], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    # Output layer
    X = Flatten()(X)
    X_output = Dense(512, activation='relu')(X)

    # merge models together to build FusionNet
    feature_merged_layer = concatenate([spectral_cnn_model_dense,
                                        X_output])

    # branch for 6 species classification
    n_final_features_species = 256
    feature_merged_dense_species = Dense(n_final_features_species, activation='relu')(feature_merged_layer)
    species_output = Dense(nspecies, activation='softmax', name="species_output")(feature_merged_dense_species)

    # branch for dead & live classification
    n_final_features_deadlive = 32
    feature_merged_dense_dl = Dense(n_final_features_deadlive,activation='relu')(feature_merged_layer)
    species_in_2 = Reshape((nspecies, 1))(species_output)
    feature_merged_dense_2_dl = Reshape((n_final_features_deadlive, 1))(feature_merged_dense_dl)
    cart_product_dl = Dot(axes=2)([feature_merged_dense_2_dl, species_in_2])
    cart_product_dl = Flatten()(cart_product_dl)
    deadlive_output = Dense(classes, activation='softmax', name="deadlive_output")(cart_product_dl)

    # here we define inputs and outputs
    data_fusion_model = Model(inputs=[spectral_cnn_model_in, lstm_shape_model_in, X_input],
                              outputs=[deadlive_output, species_output])
    opt = Adam(lr=learning_rate)
    species_weights = np.array([1, 2.5, 2, 2.5, 5, 1.5])  # 0.319466248, 0.1350, 0.1546, 0.1130, 0.0644, 0.2135
    species_loss = weighted_categorical_crossentropy(species_weights)
    losses = {
        "deadlive_output": "categorical_crossentropy",
        "species_output": species_loss  # "categorical_crossentropy"
    }
    lossWeights = {"deadlive_output": 1.0, "species_output": 0.05}
    data_fusion_model.compile(loss=losses, loss_weights=lossWeights, optimizer=opt, metrics=['acc'])

    # setting up other components for training
    fusion_model_name = f'model_save/{model_name}.h5'
    checkpointer = ModelCheckpoint(fusion_model_name, save_best_only=True,
                                   monitor='val_deadlive_output_acc', mode="max")
    plot_losses = livelossplot.PlotLossesKeras()
    reduce_lr = ReduceLROnPlateau(monitor='val_deadlive_output_loss', patience=npatience, mode='auto')
    early_stop = EarlyStopping(monitor='val_deadlive_output_loss',patience=npatience, verbose=0)
    better_progress = TqdmCallbackFix(verbose=2)
    # here we assemble
    x_train_data = [fusionnet_data.spectral.x_train, fusionnet_data.shape.x_train,
                    fusionnet_data.image.x_train]
    y_train_data = {"deadlive_output": fusionnet_data.image.y_train, 
                    "species_output": fusionnet_data.species.x_train}
    x_validation_data = [fusionnet_data.spectral.x_validation, fusionnet_data.shape.x_validation,
                         fusionnet_data.image.x_validation]
    y_validation_data = {"deadlive_output": fusionnet_data.image.y_validation, 
                         "species_output": fusionnet_data.species.x_validation}

    validation_data = (x_validation_data, y_validation_data)
    if existing_model_path is not None:
        existing_model_filename = f'{existing_model_path}/model'
        data_fusion_model = load_model(existing_model_filename,
                                         custom_objects={
                                             'mcc': mcc,
                                             'keras_better_to_categorical': keras_better_to_categorical,
                                             'keras_calculate_mcc_from_conf': keras_calculate_mcc_from_conf,
                                         })
        K.set_value(data_fusion_model.optimizer.lr, learning_rate)

    history = data_fusion_model.fit(x_train_data,
                        y_train_data,
                        batch_size=params['batch_size'],
                        epochs=epochs,
                        validation_data=validation_data, verbose=0,
                        callbacks=[checkpointer, reduce_lr, early_stop, better_progress])
    print("Net training completed")

    # Load FusionNet model with best result
    data_fusion_model.load_weights(fusion_model_name)
    loss, deadlive_loss, species_loss, validation_acc, species_acc = data_fusion_model.evaluate(x_validation_data, y_validation_data)
    print('Best validation acc of epoch:', validation_acc)
    return {"loss": -validation_acc, "status": STATUS_OK, "model": data_fusion_model, "history": history}


if __name__ == '__main__':
    # because of tensorflow.python.framework.errors_impl.InternalError:  Blas GEMM launch failed
    # import tensorflow as tf
    # physical_devices = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.config.experimental.set_memory_growth(physical_devices[1], True)
    # argments process and data setup
    args = sys.argv[1:]
    nargs = len(args)
    data_type = 0
    if nargs > 1:
        raise Exception("number of argument should be less than 2")
    elif nargs == 1:
        data_type = int(args[0])
    print(fr"culture_type: {data_type}")
    if data_type == 0:
        str_data_type = "all_species"
    elif data_type == 1:
        str_data_type = "EC"
    elif data_type == 2:
        str_data_type = "LI"
    elif data_type == 3:
        str_data_type = "SA"
    elif data_type == 4:
        str_data_type = "SM"  # all salmonella
    else:
        raise Exception(f"Wrong data_type ({data_type})")

    learning_rate = 0.001
    existing_model_path = None
    npatience = 50
    include_species = True
    model_name = f"fusion-net3_{str_data_type}"
    fusionnet_data, spec_scaler, shape_scaler, image_scaler = read_data_fusionnet_aotf_deadlive_imagescaler(data_type=data_type)

    # hyperparameters to be examined - removed most of them for final training
    hparam_space = {"batch_size": hp.choice('batch_size', [24]),
                    }
    num_evals = 1
    epochs = 1  # 10000

    # run the hyperparameter optimization
    trials = Trials()
    best_run = fmin(create_fusionnet_model,
                      hparam_space,
                      algo=tpe.suggest,
                      max_evals=num_evals,
                      trials=trials)

    # Extract and save all results
    manage_fusion_after_hyperopt_aotf_multiout(fusionnet_data, hparam_space, trials, model_name,
                                      scalers=[spec_scaler, shape_scaler, image_scaler],
                                      is_species_data=include_species)

import pathlib
import matplotlib
import matplotlib.pyplot as plt
from numpy import interp
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sns; sns.set()
import datetime
import joblib
import skimage.transform as trans
from tensorflow.keras.layers import Add, Activation, BatchNormalization, Conv2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import Regularizer
from tensorflow.keras.metrics import MeanIoU
from data_functions import *

font0=matplotlib.font_manager.FontProperties()
font0.set_style('italic')
results_folder = fr"{global_result_folder}"

# To build identity_block & convolutional_block from Rui Kang
def identity_block(X, f, filters, stage, block):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X


#  from Rui Kang
def convolutional_block(X, f, filters, stage, block, s=2):
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X


#  from Rui Kang
def convolutional_block_2(X, f, filters, stage, block, s=2):
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X_shortcut = Conv2D(filters=F2, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    return X


def draw_history_loss(history, save_dir):
    """summarize history for accuracy"""
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"{save_dir}/loss_history.pdf")
    plt.show(block=False)
    plt.pause(0.01)
    plt.close('all')


def draw_history_accuracy(history, save_dir):
    """summarize history for loss"""
    # print(f"history: {history.history}")  # fpr debugging
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'])
    else:
        plt.plot(history.history['acc'])
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'])
    else:
        plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f"{save_dir}/acc_history.pdf")
    plt.show(block=False)
    plt.pause(0.01)
    plt.close('all')


def plot_tpe_optimization(trials, model_name, save_dir, parameters=None):
    """Draw plot of TPE optimization results
    :argument:
        trials: results from hyperparameter optimization using Hyperopt
        parameters_all: parameter name list

    Example of function arguments:
        parameters = ['C', 'kernel', 'gamma']
    """
    if parameters is None:
        parameters = trials.trials[0]['misc']['vals'].keys()
    for i, val in enumerate(parameters):
        f, ax = plt.subplots(1)
        cmap = plt.cm.jet
        print(f"parameter: {i}, {val}")
        xs = [t['misc']['vals'][val] for t in trials.trials]
        xs = np.array(xs).ravel()
        ys = [-t['result']['loss'] for t in trials.trials]
        if len(xs) < 1 or len(ys) < 1:
            continue
        xs, ys = zip(*sorted(list(zip(xs, ys)), key=lambda x: x[0]))
        ys = np.array(ys)
        ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5, c=cmap(float(i)/len(parameters)))
        ax.set_title(f"model: {model_name} param: {val}")
        plt.savefig(f"{save_dir}/tpe_{val}.pdf")
        plt.show(block=False)
        plt.pause(0.01)
        plt.close('all')


def record_history(history, save_dir):
    # print(history.history.keys())
    pd.DataFrame(history.history).to_csv(f"{save_dir}/model_train_history.csv")
    # summarize history for accuracy
    draw_history_accuracy(history, save_dir)
    # summarize history for loss
    draw_history_loss(history, save_dir)


def print_hopt_space(hparam, file=None):
    print("Hyperparams testing:", file=file)
    for str in hparam:
        print(f'\t{str}:', file=file)
        print(f'{hparam[str]}', file=file)


def unlabeled_evaluate(best_model, feature_train_unlabeled, class_weight_unlabeled):
    """this function evaluates unlabeled_data.
    :param: best_model: NN model to be evaluated
    :param: feature_train_unlabeled: predictor variables
    :param: class_weight_unlabeled: npoint x nclass uncertainty matrix containing probability of each class per row
    """
    predicted = best_model.predict(feature_train_unlabeled)
    idx = np.argmax(predicted, axis=1)
    accuracy = 0
    for i,index in enumerate(idx):
        if class_weight_unlabeled[i, index] > 0:
            accuracy += 1
    accuracy /= predicted.shape[0]
    return accuracy


def manage_fusion_after_hyperopt_aotf(fusionnet_data, params, trials, model_name,
                                      is_spectral_data=True,
                                      is_shape_data=True,
                                      is_image_data=True,
                                      is_species_data=True,
                                      scalers=None):
    from scipy.io import savemat
    # get the timestamp
    str_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{results_folder}/{model_name}_{str_timestamp}"
    if not os.path.exists(save_dir):
       print("path doesn't exist. trying to make")
       os.makedirs(save_dir)
    summary_file = open(f"{save_dir}/summary.txt", "w")

    # save indice for training/valid/test
    savemat(fr'{save_dir}/index.mat',
        {
        "index_train": fusionnet_data.index_train,
        "index_vaild": fusionnet_data.index_valid,
        "index_test": fusionnet_data.index_test,

    })
    # save scalers
    if scalers is not None:
        if len(scalers) == 3:
            spec_scaler, shape_scaler, image_scaler = scalers
            from joblib import dump, load
            dump(spec_scaler, fr'{save_dir}/spec_scaler.bin')
            dump(shape_scaler, fr'{save_dir}/shape_scaler.bin')
            dump(image_scaler, fr'{save_dir}/image_scaler.bin')
        else:
            spec_scaler, shape_scaler = scalers
            from joblib import dump, load
            dump(spec_scaler, fr'{save_dir}/spec_scaler.bin')
            dump(shape_scaler, fr'{save_dir}/shape_scaler.bin')

    data_exist = [is_spectral_data, is_shape_data, is_image_data, True]  # is_species_data]
    feature_train = [fusionnet_data.spectral.x_train, fusionnet_data.shape.x_train,
                     fusionnet_data.image.x_train, fusionnet_data.species.x_train]
    feature_validation = [fusionnet_data.spectral.x_validation, fusionnet_data.shape.x_validation,
                          fusionnet_data.image.x_validation, fusionnet_data.species.x_validation]
    feature_test = [fusionnet_data.spectral.x_test, fusionnet_data.shape.x_test,
                    fusionnet_data.image.x_test, fusionnet_data.species.x_test]
    feature_train = [d for (d, exist) in zip(feature_train, data_exist) if exist]
    feature_validation = [d for (d, exist) in zip(feature_validation, data_exist) if exist]
    feature_test = [d for (d, exist) in zip(feature_test, data_exist) if exist]
    if hasattr('fusionnet_data.spectral', 'x_train_unlabeled'):
        feature_train_unlabeled = [fusionnet_data.spectral.x_train_unlabeled,
                                   fusionnet_data.shape.x_train_unlabeled,
                                   fusionnet_data.image.x_train_unlabeled,
                                   fusionnet_data.species.x_train_unlabeled]
        feature_test_unlabeled = [fusionnet_data.spectral.x_test_unlabeled,
                                   fusionnet_data.shape.x_test_unlabeled,
                                   fusionnet_data.image.x_test_unlabeled,
                                  fusionnet_data.species.x_test_unlabeled]

        feature_train_unlabeled = [d for (d, exist) in zip(feature_train_unlabeled, data_exist) if exist]
        feature_test_unlabeled = [d for (d, exist) in zip(feature_test_unlabeled, data_exist) if exist]

    y_validation = fusionnet_data.image.y_validation
    y_test = fusionnet_data.image.y_test
    y_train = fusionnet_data.image.y_train
    if hasattr('fusionnet_data.spectral', 'x_train_unlabeled'):
        y_train_unlabeled = fusionnet_data.image.y_train_unlabeled
        y_test_unlabeled = fusionnet_data.image.y_test_unlabeled

    # print summary into summary file
    print_hopt_space(params, summary_file)
    print("Evalutation of best performing model:", file=summary_file)
    best_trial = trials.best_trial
    best_model = best_trial['result']['model']
    print(best_model.history.history.keys())
    print(best_model.evaluate(feature_train, y_train), file=summary_file)
    print(best_model.evaluate(feature_validation, y_validation), file=summary_file)
    print(best_model.evaluate(feature_test, y_test), file=summary_file)
    if hasattr('fusionnet_data.spectral', 'x_train_unlabeled'):
        print(unlabeled_evaluate(best_model, feature_train_unlabeled, y_train_unlabeled), file=summary_file)
        print(unlabeled_evaluate(best_model, feature_test_unlabeled, y_test_unlabeled), file=summary_file)
    print("Best performing model chosen hyper-parameters:", file=summary_file)
    print(best_trial['misc']['vals'], file=summary_file)

    # print to screen
    print_hopt_space(params)
    print("Evalutation of best performing model: ")
    print(best_model.evaluate(feature_train, y_train))
    print(best_model.evaluate(feature_validation, y_validation))
    print(best_model.evaluate(feature_test, y_test))
    if hasattr('fusionnet_data.spectral', 'x_train_unlabeled'):
        print(unlabeled_evaluate(best_model, feature_train_unlabeled, y_train_unlabeled))
        print(unlabeled_evaluate(best_model, feature_test_unlabeled, y_test_unlabeled))
    print("Best performing model chosen hyper-parameters:")
    print(best_trial['misc']['vals'])

    best_model.save(f"{save_dir}/model")
    # serialize weights to HDF5
    best_model.save_weights(f"{save_dir}/model_weights.h5")
    print("Saved model to disk")
    # plot & save history for best model
    if best_model.history.history:
        history = best_model.history
    else:
        history = best_trial['result']['history']
    record_history(history, save_dir)
    # plot trials
    plot_tpe_optimization(trials, model_name, save_dir)
    summary_file.close()


def manage_fusion_after_hyperopt_aotf_multiout(fusionnet_data, params, trials, model_name,
                                      is_spectral_data=True,
                                      is_shape_data=True,
                                      is_image_data=True,
                                      is_species_data=True,
                                      scalers=None):
    from scipy.io import savemat

    # get the timestamp
    str_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{results_folder}/{model_name}_{str_timestamp}"
    if not os.path.exists(save_dir):
       print("path doesn't exist. trying to make")
       os.makedirs(save_dir)
    summary_file = open(f"{save_dir}/summary.txt", "w")

    # save indice for training/valid/test
    savemat(fr'{save_dir}/index.mat',
        {
        "index_train": fusionnet_data.index_train,
        "index_vaild": fusionnet_data.index_valid,
        "index_test": fusionnet_data.index_test,

    })

    # save scalers
    if scalers is not None:
        if len(scalers) == 3:
            spec_scaler, shape_scaler, image_scaler = scalers
            from joblib import dump, load
            dump(spec_scaler, fr'{save_dir}/spec_scaler.bin')
            dump(shape_scaler, fr'{save_dir}/shape_scaler.bin')
            dump(image_scaler, fr'{save_dir}/image_scaler.bin')
        else:
            spec_scaler, shape_scaler = scalers
            from joblib import dump, load
            dump(spec_scaler, fr'{save_dir}/spec_scaler.bin')
            dump(shape_scaler, fr'{save_dir}/shape_scaler.bin')

    data_exist = [is_spectral_data, is_shape_data, is_image_data, True]  # is_species_data]
    feature_train = [fusionnet_data.spectral.x_train, fusionnet_data.shape.x_train,
                     fusionnet_data.image.x_train]
    feature_validation = [fusionnet_data.spectral.x_validation, fusionnet_data.shape.x_validation,
                          fusionnet_data.image.x_validation]
    feature_test = [fusionnet_data.spectral.x_test, fusionnet_data.shape.x_test,
                    fusionnet_data.image.x_test]
    feature_train = [d for (d, exist) in zip(feature_train, data_exist) if exist]
    feature_validation = [d for (d, exist) in zip(feature_validation, data_exist) if exist]
    feature_test = [d for (d, exist) in zip(feature_test, data_exist) if exist]
    if hasattr('fusionnet_data.spectral', 'x_train_unlabeled'):
        feature_train_unlabeled = [fusionnet_data.spectral.x_train_unlabeled,
                                   fusionnet_data.shape.x_train_unlabeled,
                                   fusionnet_data.image.x_train_unlabeled]
        feature_test_unlabeled = [fusionnet_data.spectral.x_test_unlabeled,
                                   fusionnet_data.shape.x_test_unlabeled,
                                   fusionnet_data.image.x_test_unlabeled]

        feature_train_unlabeled = [d for (d, exist) in zip(feature_train_unlabeled, data_exist) if exist]
        feature_test_unlabeled = [d for (d, exist) in zip(feature_test_unlabeled, data_exist) if exist]

    y_validation = [fusionnet_data.image.y_validation, fusionnet_data.species.x_validation]
    y_test = [fusionnet_data.image.y_test, fusionnet_data.species.x_test]
    y_train = [fusionnet_data.image.y_train, fusionnet_data.species.x_train]
    if hasattr('fusionnet_data.spectral', 'x_train_unlabeled'):
        y_train_unlabeled = fusionnet_data.image.y_train_unlabeled
        y_test_unlabeled = fusionnet_data.image.y_test_unlabeled
    # print summary into summary file
    print_hopt_space(params, summary_file)
    print("Evalutation of best performing model:", file=summary_file)
    best_trial = trials.best_trial
    best_model = best_trial['result']['model']
    print(best_model.history.history.keys())
    print(best_model.evaluate(feature_train, y_train), file=summary_file)
    print(best_model.evaluate(feature_validation, y_validation), file=summary_file)
    print(best_model.evaluate(feature_test, y_test), file=summary_file)
    if hasattr('fusionnet_data.spectral', 'x_train_unlabeled'):
        print(unlabeled_evaluate(best_model, feature_train_unlabeled, y_train_unlabeled), file=summary_file)
        print(unlabeled_evaluate(best_model, feature_test_unlabeled, y_test_unlabeled), file=summary_file)
    print("Best performing model chosen hyper-parameters:", file=summary_file)
    print(best_trial['misc']['vals'], file=summary_file)

    # print to screen
    print_hopt_space(params)
    print("Evalutation of best performing model: ")
    print(best_model.evaluate(feature_train, y_train))
    print(best_model.evaluate(feature_validation, y_validation))
    print(best_model.evaluate(feature_test, y_test))
    if hasattr('fusionnet_data.spectral', 'x_train_unlabeled'):
        print(unlabeled_evaluate(best_model, feature_train_unlabeled, y_train_unlabeled))
        print(unlabeled_evaluate(best_model, feature_test_unlabeled, y_test_unlabeled))
    print("Best performing model chosen hyper-parameters:")
    print(best_trial['misc']['vals'])

    best_model.save(f"{save_dir}/model")
    # serialize weights to HDF5
    best_model.save_weights(f"{save_dir}/model_weights.h5")
    print("Saved model to disk")
    # plot & save history for best model
    if best_model.history.history:
        history = best_model.history
    else:
        history = best_trial['result']['history']
    # plot trials
    plot_tpe_optimization(trials, model_name, save_dir)
    summary_file.close()


def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

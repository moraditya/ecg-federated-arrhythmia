import tensorflow as tf
import tensorflow_privacy as tfp


def build_ecg_model(input_shape = (360,1), num_classes = 5):
    """ 
    Builds a production-ready convolutional neural network (CNN) model for ECG arrhythmia detection with differential privacy.
    """

    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32,5,activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(64,5,activation = 'relu'),
        tf.keras.layers.MaxPooling1D(pool_size = 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation = 'relu'),
        tf.keras.layers.Dense(num_classes, activation = 'softmax')
    ])

    # add a DP optimizer
    optimizer = tfp.DPKerasSGDOptimizer(
        l2_norm_clip = 1.5,
        noise_multiplier = 1.3,
        num_microbatches = 250,
        learning_rate = 0.25
    )


    model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return model


if __name__ == '__main__':
    model = build_ecg_model()
    model.summary()
    
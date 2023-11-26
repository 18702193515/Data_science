import tensorflow as tf

def GetModel_V1(input_shape, class_num, is_training=True):
    input_ = tf.keras.Input(shape=input_shape)

    conv1 = tf.keras.layers.Conv2D(64, 7, 2, 'SAME', activation='relu')(input_)
    pool1 = tf.keras.layers.MaxPooling2D()(conv1)
    conv2 = tf.keras.layers.Conv2D(256, 3, 2, 'SAME', activation='relu')(pool1)
    pool2 = tf.keras.layers.MaxPooling2D()(conv2)
    conv3 = tf.keras.layers.Conv2D(512, 3, 2, 'SAME', activation='relu')(pool2)
    pool3 = tf.keras.layers.AveragePooling2D()(conv3)

    flat = tf.keras.layers.Flatten()(pool3)
    output = tf.keras.layers.Dense(class_num, activation='softmax')(flat)

    model = tf.keras.Model(inputs=input_, outputs=output)
    model.summary()  # 将模型信息输出到终端
    return model


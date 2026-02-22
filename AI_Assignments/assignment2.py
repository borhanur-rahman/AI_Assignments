from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model



inputs = Input(shape=(28, 28), name='input_layer')

x = Dense(8, activation='relu', name='hidden1')(inputs)
x = Dense(16, activation='relu', name='hidden2')(x)
x = Dense(32, activation='relu', name='hidden3')(x)
x = Dense(64, activation='relu', name='hidden4')(x)
x = Dense(128, activation='relu', name='hidden5')(x)
x = Dense(256, activation='relu', name='hidden6')(x)
x = Dense(512, activation='relu', name='hidden7')(x)
x = Dense(32, activation='relu', name='hidden8')(x)
x = Dense(16, activation='relu', name='hidden9')(x)

outputs = Dense(10, activation='softmax', name='output_layer')(x)
model = Model(inputs=inputs, outputs=outputs, name="Deep_FCFNN_Model")


model.summary(show_trainable=True)


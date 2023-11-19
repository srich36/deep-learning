import tensorflow as tf
import numpy as np
from keras import Sequential
from keras.layers import Dense, Input
from keras.src.callbacks import ModelCheckpoint

checkpoint_filepath = 'training/checkpoint.ckpt'
data_file = 'data.txt'


def get_model(batch_size: int) -> Sequential:
    """
    The model is a simple feedforward neural network with 2 inputs, 1 hidden layer with 24 neurons, and 1 output.
    - The input layer is 2 features
    - The hidden layer is 24 neurons with sigmoid activation
    - The output layer is 1 neuron with sigmoid activation
        - This is for a binary classification
    """
    model = Sequential([
        Input(shape=(2,), batch_size=batch_size, name='input_layer'),
        Dense(24, activation='sigmoid', name='hidden_layer'),
        Dense(1, activation='sigmoid', name='output_layer')
    ])
    return model


def predict():

    data = [
        (0, 0, False),
        (10, 100, True),
        (8, 30, False),
        (5, 50, False),
        (5, 80, True),
        (6, 70, True),
        (4, 75, False),
        (9, 35, True),
        (12, 85, True),
        (1, 15, True)
    ]
    model = get_model(batch_size=1)
    model.load_weights(checkpoint_filepath)

    for hours, attendance, should_pass in data:
        data = np.array([hours, attendance])

        # Model inputs should be an array of input vectors
        # The model expects inputs in batches, which is why we need an array of arrays
        data = data.reshape(1, 2)
        res = model.predict(data, verbose=0)
        probability = res[0][0]

        def is_correct_prediction():
            if should_pass:
                return probability > 0.85
            return probability < 0.15

        print(f'Hours: {hours}, Attendance: {attendance}, Should Pass: {should_pass} -- Predicted probability: {probability:.0%} -- Correct: {is_correct_prediction()}')



def get_training_data() -> (np.ndarray, np.ndarray):
    with open(data_file) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    x_data = []
    y_data = []
    for line in lines:
        hours, attendance, label = line.split(',')
        x_data.append([int(hours), float(attendance)])
        y_data.append(int(label))
    return np.array(x_data), np.array(y_data)


def train_model():

    batch_size = 64
    training_checkpoint = ModelCheckpoint(
        checkpoint_filepath,
        save_weights_only=True,
        verbose=1
    )
    x_train, y_train = get_training_data()

    model = get_model(batch_size=batch_size)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(
        x_train,
        y_train,
        callbacks=[training_checkpoint],
        batch_size=batch_size,
        epochs=11
    )


if __name__ == '__main__':
    # train_model()
    predict()

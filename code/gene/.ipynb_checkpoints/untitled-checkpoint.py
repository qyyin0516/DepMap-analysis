import numpy as np
import tensorflow as tf
import math
import random

tf.config.experimental_run_functions_eagerly(True)
random.seed(2)
np.random.seed(2)
tf.random.set_seed(2)


class RegularizationLayer(tf.keras.layers.Layer):
    def __init__(self, reg_type, alpha_reg, alpha_binomial, n, p, **kwargs):
        super(RegularizationLayer, self).__init__(**kwargs)
        self.reg_type = reg_type
        self.alpha_reg = alpha_reg
        self.alpha_binomial = alpha_binomial
        self.n = n
        self.p = p

    def call(self, inputs):
        rounded = tf.math.round(tf.reduce_sum(inputs, axis=0))
        mapped = tf.map_fn(lambda k: truncated_binomial_log_pmf(k, self.n, self.p), rounded, dtype=tf.float32)
        distribution_loss = -tf.reduce_mean(mapped)
        if self.reg_type == 'L2':
            regularization_loss = tf.reduce_mean(tf.square(inputs))
        elif self.reg_type == 'L1':
            regularization_loss = tf.reduce_mean(tf.abs(inputs))
        self.add_loss(self.alpha_reg * regularization_loss + self.alpha_binomial * distribution_loss)
        return inputs


class Encoder(tf.keras.layers.Layer):
    def __init__(self, masking):
        super(Encoder, self).__init__()
        self.masking = masking
        self.dense_layers = [tf.keras.layers.Dense(units, activation=tf.nn.tanh) for units in encoder_layer_sizes[:-1]]
        self.regularization_layer = RegularizationLayer(reg_type=reg_type, alpha_reg=alpha_reg, alpha_binomial=alpha_binomial, n=batch_size, p=estimated_p)

        self.weight_variables = [tf.Variable(tf.random.normal([encoder_layer_sizes[i - 1], encoder_layer_sizes[i]])) for
                                 i in range(1, len(encoder_layer_sizes))]
        self.bias_variables = [tf.Variable(tf.random.normal([encoder_layer_sizes[i]])) for i in
                               range(1, len(encoder_layer_sizes))]

    def call(self, inputs):
        x = inputs
        for i, layer in enumerate(self.dense_layers):
            if i == 3:
                x = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weight_variables[i]), self.bias_variables[i]))
                x = self.regularization_layer(x)
                # x = tf.round(x)
            else:
                # masked_weights = tf.multiply(self.weight_variables[i], self.masking[len(self.masking) - i - 1].T)
                x = tf.nn.tanh(tf.add(tf.matmul(x, self.weight_variables[i]), self.bias_variables[i]))
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, masking):
        super(Decoder, self).__init__()
        self.masking = masking
        self.dense_layers = [tf.keras.layers.Dense(units, activation=tf.nn.tanh) for units in decoder_layer_sizes[:-1]]

        self.weight_variables = [tf.Variable(tf.random.normal([decoder_layer_sizes[i], decoder_layer_sizes[i + 1]])) for
                                 i in range(len(decoder_layer_sizes) - 1)]
        self.bias_variables = [tf.Variable(tf.random.normal([decoder_layer_sizes[i + 1]])) for i in
                               range(len(decoder_layer_sizes) - 1)]

    def call(self, inputs):
        x = inputs
        for i, layer in enumerate(self.dense_layers):
            if i == 0:
                x = tf.nn.tanh(tf.add(tf.matmul(x, self.weight_variables[i]), self.bias_variables[i]))
            else:
                masked_weights = tf.multiply(self.weight_variables[i], self.masking[i - 1])
                x = tf.nn.tanh(tf.add(tf.matmul(x, masked_weights), self.bias_variables[i]))
        return x


class Autoencoder(tf.keras.Model):
    def __init__(self, encoder, decoder, masking):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.masking = masking

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


encoder = Encoder(masking)
decoder = Decoder(masking)
autoencoder = Autoencoder(encoder, decoder, masking)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


@tf.function
def train_step(inputs):
    with tf.GradientTape() as tape:
        predictions = autoencoder(inputs)
        reconstruction_loss = tf.keras.losses.mean_squared_error(inputs, predictions)
        regularization_loss = sum(autoencoder.encoder.losses)  # + sum(autoencoder.decoder.losses)
        total_loss = reconstruction_loss + regularization_loss
    gradients = tape.gradient(total_loss, autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
    return tf.reduce_mean(total_loss).numpy()


for epoch in range(num_epochs):
    if epoch != 0:
        prev_weight = copy.deepcopy(autoencoder.weights)
    num_samples = len(dt_np)
    num_batches = math.ceil(num_samples / batch_size)

    total_loss = 0.
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)
        batch_x = dt_np[start_idx:end_idx]
        loss = train_step(batch_x)
        total_loss += loss
    avg_loss = total_loss / num_batches
    if avg_loss == np.Inf:
        break
    print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_loss))
if epoch == num_epochs - 1:
    prev_weight = copy.deepcopy(autoencoder.weights)
autoencoder.set_weights(prev_weight)
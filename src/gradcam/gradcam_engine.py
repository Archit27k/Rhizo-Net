import tensorflow as tf
import numpy as np
import cv2

class GradCAM:

    def __init__(self, model):
        self.model = model

    def generate(self, image_array, layer_name):

        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [
                self.model.get_layer(layer_name).output,
                self.model.output
            ]
        )

        with tf.GradientTape() as tape:

            conv_outputs, predictions = grad_model(image_array)

            loss = predictions[:, np.argmax(predictions[0])]

        grads = tape.gradient(loss, conv_outputs)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

        heatmap = tf.squeeze(heatmap)

        heatmap = np.maximum(heatmap, 0)

        heatmap /= np.max(heatmap)

        return heatmap.numpy()
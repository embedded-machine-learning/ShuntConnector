import tensorflow as tf

from tensorflow.keras.layers import DepthwiseConv2D

def find_input_output_dilation_rates(model, shunt_locations):

    dilation_rates = []

    for i, layer in enumerate(model.layers):
        if shunt_locations[0] <= i <= shunt_locations[1]:
            if isinstance(layer, DepthwiseConv2D):
                config = layer.get_config()
                dilation_rates.append(config['dilation_rate'][0])

    dilation_rate_input = dilation_rates[0]
    dilation_rate_output = dilation_rates[-1]

    return dilation_rate_input, dilation_rate_output
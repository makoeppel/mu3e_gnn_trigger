import keras
from qkeras import QConv2D, QDense, QActivation, quantized_bits



def create_cnn_model(max_hits=256, num_channels=[8, 16, 32], dense_dim=32, input_quant='quantized_bits(8,3)', use_time=True):
    q_bits = quantized_bits(8, 3, alpha=1)

    if use_time: inputs = keras.layers.Input(shape=(1, max_hits, 4))
    if not use_time: inputs = keras.layers.Input(shape=(1, max_hits, 3))
    x = QActivation(input_quant, name='q_input_activation')(inputs)

    for i, filters in enumerate(num_channels):
        layer_idx = i + 1
        x = QConv2D(
            filters=filters, 
            kernel_size=(1, 5), 
            padding='same',
            kernel_quantizer=q_bits,
            bias_quantizer=q_bits,
            name=f'q_conv{layer_idx}'
        )(x)
        x = QActivation('quantized_relu(8, 3)', name=f'q_relu{layer_idx}')(x)
        x = keras.layers.MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid', name=f'q_pool{layer_idx}')(x)

    x = keras.layers.Flatten()(x)
    x = QDense(dense_dim, kernel_quantizer=q_bits, bias_quantizer=q_bits, name='q_dense1')(x)
    x = QActivation('quantized_relu(8, 3)', name=f'q_relu_dense')(x)

    outputs = QDense(1, kernel_quantizer=q_bits, bias_quantizer=q_bits, activation='sigmoid', name='q_output')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name=f"Quantized_Spacetime_{len(num_channels)}Layer_CNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.002),
        loss='binary_crossentropy',
        metrics=['AUC']
    )

    model.summary()
    total_params = model.count_params()

    return model, total_params

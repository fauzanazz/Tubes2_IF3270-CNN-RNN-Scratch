import numpy as np

def predict(network, inputs, batch_size=32):
    outputs = []

    for i in range(0, len(inputs), batch_size):
        print("mulai prediksi batch ke-",i//batch_size,"dari",len(inputs)//batch_size)
        batch = inputs[i:i+batch_size]
        out = batch
        for layer in network:
            out = layer.forward(out)
        outputs.append(out)  # out sudah bentuk batch

    return np.concatenate(outputs, axis=0)  # gabungkan semua batch

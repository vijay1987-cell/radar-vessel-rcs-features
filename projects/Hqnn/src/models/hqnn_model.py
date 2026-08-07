import numpy as np
import torch
import torch.nn as nn
import pennylane as qml


def _build_device(n_qubits: int):
    import os
    if os.environ.get('PENNYLANE_USE_GPU', '0') == '1':
        try:
            dev = qml.device('lightning.gpu', wires=n_qubits)
            return dev
        except Exception:
            pass
    try:
        return qml.device('lightning.qubit', wires=n_qubits)
    except Exception:
        return qml.device('default.qubit', wires=n_qubits)


class HQNNClassifier(nn.Module):
    """
    Hybrid Quantum-Classical Neural Network.
    Architecture:
        Classical encoder (dense layers)
        → scale to [-π,π]
        → Quantum circuit (AngleEmbedding + ansatz)
        → Classical decoder (dense layers)
        → Softmax
    """

    def __init__(self, n_features: int, n_qubits: int, n_layers: int, n_classes: int,
                 classical_hidden: int = 16, n_classical_layers: int = 1,
                 ansatz: str = 'strongly_entangling', activation: str = 'relu'):
        super().__init__()
        self.n_qubits = n_qubits

        act_fn = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'leaky_relu': nn.LeakyReLU}
        act = act_fn.get(activation, nn.ReLU)

        # Classical encoder
        encoder_layers = []
        in_size = n_features
        for _ in range(n_classical_layers):
            encoder_layers += [nn.Linear(in_size, classical_hidden), act()]
            in_size = classical_hidden
        encoder_layers += [nn.Linear(in_size, n_qubits), nn.Tanh()]
        self.encoder = nn.Sequential(*encoder_layers)

        # Quantum layer
        dev = _build_device(n_qubits)
        diff = 'adjoint' if 'lightning' in dev.name else 'backprop'

        @qml.qnode(dev, interface='torch', diff_method=diff)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            if ansatz == 'strongly_entangling':
                qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            elif ansatz == 'basic_entangler':
                qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            elif ansatz == 'hardware_efficient':
                for layer_idx in range(weights.shape[0]):
                    for i in range(n_qubits):
                        qml.RY(weights[layer_idx, i, 0], wires=i)
                        qml.RZ(weights[layer_idx, i, 1], wires=i)
                    for i in range(0, n_qubits - 1, 2):
                        qml.CNOT(wires=[i, i + 1])
                    for i in range(1, n_qubits - 1, 2):
                        qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        if ansatz == 'strongly_entangling':
            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        elif ansatz == 'hardware_efficient':
            weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        else:
            weight_shapes = {"weights": (n_layers, n_qubits)}

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Classical decoder
        decoder_layers = []
        in_size = n_qubits
        for _ in range(n_classical_layers):
            decoder_layers += [nn.Linear(in_size, classical_hidden), act()]
            in_size = classical_hidden
        decoder_layers.append(nn.Linear(in_size, n_classes))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x) * np.pi
        x = self.qlayer(x)
        return self.decoder(x)

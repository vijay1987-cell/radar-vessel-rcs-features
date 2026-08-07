import numpy as np
import torch
import torch.nn as nn
import pennylane as qml


def _build_device(n_qubits: int):
    try:
        return qml.device('lightning.qubit', wires=n_qubits)
    except Exception:
        return qml.device('default.qubit', wires=n_qubits)


def _make_circuit(dev, n_qubits: int, n_layers: int, embedding: str, ansatz: str):
    @qml.qnode(dev, interface='torch', diff_method='adjoint' if 'lightning' in dev.name else 'backprop')
    def circuit(inputs, weights):
        if embedding == 'angle':
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
        elif embedding == 'zzfeaturemap':
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(2.0 * inputs[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(2.0 * (np.pi - inputs[i]) * (np.pi - inputs[i + 1]), wires=i + 1)
                qml.CNOT(wires=[i, i + 1])

        if ansatz == 'strongly_entangling':
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        elif ansatz == 'basic_entangler':
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        elif ansatz == 'hardware_efficient':
            for layer in range(weights.shape[0]):
                for i in range(n_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                for i in range(0, n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(1, n_qubits - 1, 2):
                    qml.CNOT(wires=[i, i + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class VQCClassifier(nn.Module):
    """
    Variational Quantum Classifier.
    Architecture: Linear projection → scale to [-π,π] → Quantum circuit → Linear output → Softmax
    """

    def __init__(self, n_features: int, n_qubits: int, n_layers: int, n_classes: int,
                 embedding: str = 'angle', ansatz: str = 'strongly_entangling'):
        super().__init__()
        self.n_qubits = n_qubits

        # Classical pre-processing layer to map n_features → n_qubits
        self.pre = nn.Sequential(
            nn.Linear(n_features, n_qubits),
            nn.Tanh(),  # output in [-1,1]; scaled to [-π,π] in forward
        )

        dev = _build_device(n_qubits)
        circuit = _make_circuit(dev, n_qubits, n_layers, embedding, ansatz)

        if ansatz == 'strongly_entangling':
            weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        elif ansatz == 'hardware_efficient':
            weight_shapes = {"weights": (n_layers, n_qubits, 2)}
        else:
            weight_shapes = {"weights": (n_layers, n_qubits)}

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.output = nn.Linear(n_qubits, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x) * np.pi
        x = self.qlayer(x)
        return self.output(x)

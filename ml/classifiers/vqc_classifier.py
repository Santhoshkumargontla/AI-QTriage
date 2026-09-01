"""Experimental 4-qubit VQC. Training and inference share one QNode. No fabricated outputs."""
import os
import pickle

import numpy as np
import pennylane as qml

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ml.models.canonical_paths import resolve_existing

MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
EXPERIMENTAL_ONLY = "EXPERIMENTAL_ONLY"

CIRCUIT_SPEC = {
    "embedding": "AngleEmbedding",
    "rotation": "X",
    "ansatz": "StronglyEntanglingLayers",
    "num_qubits": 4,
    "num_layers": 2,
    "measurement": "PauliZ_expval_per_qubit",
    "postprocess": "linear_4_to_3_softmax",
    "device": "default.qubit",
}


class VQCClassifier:
    """Variational Quantum Classifier (VQC) with classical StandardScaler and PCA projection."""

    def __init__(self, model_dir: str = None):
        self.num_qubits = CIRCUIT_SPEC["num_qubits"]
        self.num_layers = CIRCUIT_SPEC["num_layers"]
        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.num_qubits)
        rng = np.random.RandomState(42)
        self.q_weights = rng.randn(self.num_layers, self.num_qubits, 3) * 0.1
        self.lin_weights = rng.randn(3, self.num_qubits) * 0.1
        self.lin_bias = np.zeros(3)
        self.is_trained = False
        self.status = "UNINITIALIZED"
        self.model_dir = model_dir
        self.loss_history = []
        self.optimization = None
        self.circuit = qml.QNode(self._vqc_circuit, self.dev)
        if model_dir:
            self.load_model(resolve_existing(model_dir))

    def _vqc_circuit(self, weights, features):
        """Angle Embedding followed by Strongly Entangling variational layers.

        This is the single circuit used by both train() and predict().
        """
        qml.AngleEmbedding(features, wires=range(self.num_qubits), rotation="X")
        qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def circuit_spec(self) -> dict:
        return {
            **CIRCUIT_SPEC,
            "qnode": "VQCClassifier._vqc_circuit",
            "train_forward": "VQCClassifier._forward",
            "predict_forward": "VQCClassifier._forward",
            "circuits_match": True,
        }

    def _forward(self, q_w, lin_w, lin_b, x_pca):
        """Shared forward pass for train and predict. Keep qml.math so Adam can differentiate the QNode."""
        expvals = qml.math.stack(self.circuit(q_w, x_pca))
        scores = qml.math.dot(lin_w, expvals) + lin_b
        scores = scores - qml.math.max(scores)
        exp_scores = qml.math.exp(scores)
        return exp_scores / qml.math.sum(exp_scores)

    def _unavailable(self, extra: str) -> RuntimeError:
        return RuntimeError(f"{MODEL_UNAVAILABLE}: {extra}")

    def load_model(self, model_dir: str):
        """Loads scaler, PCA, and VQC weights from disk. Never fabricates missing tensors."""
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        pca_path = os.path.join(model_dir, "pca.pkl")
        vqc_path = os.path.join(model_dir, "vqc_weights.npz")
        missing = [p for p in (scaler_path, pca_path, vqc_path) if not os.path.exists(p)]
        if missing:
            self.status = MODEL_UNAVAILABLE
            self.is_trained = False
            raise FileNotFoundError(f"{MODEL_UNAVAILABLE}: missing {', '.join(missing)}")

        with open(scaler_path, "rb") as handle:
            self.scaler = pickle.load(handle)
        with open(pca_path, "rb") as handle:
            self.pca = pickle.load(handle)

        weights = np.load(vqc_path)
        required = ("q_weights", "lin_weights", "lin_bias")
        absent = [name for name in required if name not in weights.files]
        if absent:
            self.status = MODEL_UNAVAILABLE
            self.is_trained = False
            raise RuntimeError(f"{MODEL_UNAVAILABLE}: weight file missing keys {absent}")

        self.q_weights = np.array(weights["q_weights"])
        self.lin_weights = np.array(weights["lin_weights"])
        self.lin_bias = np.array(weights["lin_bias"])
        if self.q_weights.shape != (self.num_layers, self.num_qubits, 3):
            self.status = MODEL_UNAVAILABLE
            self.is_trained = False
            raise RuntimeError(
                f"{MODEL_UNAVAILABLE}: q_weights shape {self.q_weights.shape} "
                f"!= {(self.num_layers, self.num_qubits, 3)}"
            )
        if self.lin_weights.shape != (3, self.num_qubits) or self.lin_bias.shape != (3,):
            self.status = MODEL_UNAVAILABLE
            self.is_trained = False
            raise RuntimeError(f"{MODEL_UNAVAILABLE}: linear head shape mismatch")

        self.model_dir = model_dir
        self.is_trained = True
        self.status = EXPERIMENTAL_ONLY
        print(f"VQC model successfully loaded from {model_dir}")

    def save_model(self, model_dir: str):
        """Saves scaler, PCA, and VQC weights to disk."""
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "scaler.pkl"), "wb") as handle:
            pickle.dump(self.scaler, handle)
        with open(os.path.join(model_dir, "pca.pkl"), "wb") as handle:
            pickle.dump(self.pca, handle)
        np.savez(
            os.path.join(model_dir, "vqc_weights.npz"),
            q_weights=self.q_weights,
            lin_weights=self.lin_weights,
            lin_bias=self.lin_bias,
        )
        print(f"VQC model successfully saved to {model_dir}")

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 15, lr: float = 0.2):
        """
        Fits StandardScaler + PCA on TRAIN only, then optimizes the shared VQC circuit.
        Uses PennyLane Adam on the same QNode as predict(). Records a loss curve.
        """
        print("Executing classical scaling and PCA-based dimensionality reduction...")
        X_scaled = self.scaler.fit_transform(X)
        X_pca = np.clip(self.pca.fit_transform(X_scaled), -3.0, 3.0)

        from pennylane import numpy as pnp

        q_w = pnp.array(np.array(self.q_weights, dtype=float), requires_grad=True)
        lin_w = pnp.array(np.array(self.lin_weights, dtype=float), requires_grad=True)
        lin_b = pnp.array(np.array(self.lin_bias, dtype=float), requires_grad=True)
        y_onehot = np.zeros((len(y), 3))
        y_onehot[np.arange(len(y)), np.asarray(y, dtype=int)] = 1.0

        def cost(q_w, lin_w, lin_b):
            loss = pnp.array(0.0)
            for i in range(len(y)):
                probs = self._forward(q_w, lin_w, lin_b, X_pca[i])
                loss = loss - pnp.sum(y_onehot[i] * pnp.log(probs + 1e-15))
            return loss / len(y)

        opt = qml.AdamOptimizer(stepsize=lr)
        self.loss_history = []
        print("Fitting Variational Quantum Circuit parameters (Adam on shared QNode)...")
        for epoch in range(epochs):
            (q_w, lin_w, lin_b), current_loss = opt.step_and_cost(cost, q_w, lin_w, lin_b)
            loss_f = float(current_loss)
            self.loss_history.append(loss_f)
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"  VQC Epoch {epoch + 1}/{epochs} | Loss: {loss_f:.4f}")

        self.q_weights = np.array(q_w)
        self.lin_weights = np.array(lin_w)
        self.lin_bias = np.array(lin_b)
        self.is_trained = True
        self.status = EXPERIMENTAL_ONLY
        self.optimization = {
            "optimizer": "PennyLane AdamOptimizer",
            "epochs": int(epochs),
            "lr": float(lr),
            "train_samples": int(len(y)),
            "loss_start": self.loss_history[0] if self.loss_history else None,
            "loss_end": self.loss_history[-1] if self.loss_history else None,
            "loss_decreased": (
                bool(self.loss_history[-1] < self.loss_history[0])
                if len(self.loss_history) >= 2
                else False
            ),
            "circuit": self.circuit_spec(),
        }
        print("VQC Classifier training completed successfully.")

    def predict(self, fused_vector: np.ndarray) -> tuple:
        """
        Projects input through train-fitted scaler/PCA and classifies with the shared QNode.
        Never returns a hardcoded class or fabricated probability vector.
        """
        if not self.is_trained:
            raise self._unavailable("VQC classifier is not trained or weights were not loaded.")

        x_in = fused_vector.reshape(1, -1)
        expected = int(getattr(self.scaler, "n_features_in_", x_in.shape[1]))
        if x_in.shape[1] != expected:
            raise ValueError(
                f"{MODEL_UNAVAILABLE}: VQC feature dimension mismatch: expected {expected}, got {x_in.shape[1]}"
            )
        x_scaled = self.scaler.transform(x_in)
        x_pca = np.clip(self.pca.transform(x_scaled)[0], -3.0, 3.0)
        probs = np.asarray(self._forward(self.q_weights, self.lin_weights, self.lin_bias, x_pca), dtype=float)

        if probs.shape != (3,) or not np.all(np.isfinite(probs)):
            raise self._unavailable(f"non-finite or malformed probabilities: {probs}")
        if abs(float(np.sum(probs)) - 1.0) > 1e-4:
            raise self._unavailable(f"probabilities do not sum to 1: {probs}")

        pred_idx = int(np.argmax(probs))
        return pred_idx, probs.tolist()

    def get_info(self) -> dict:
        return {
            "status": self.status,
            "is_trained": self.is_trained,
            "experimental_only": True,
            "used_in_main_decision": False,
            "model_dir": self.model_dir,
            "data_provenance": "SYNTHETIC",
            "circuit": self.circuit_spec(),
            "optimization": self.optimization,
        }

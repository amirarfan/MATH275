# Authors:
# Denny Mattern (denny.mattern@fokus.fraunhofer.de)
# Darya Martyniuk (darya.martyniuk@fokus.fraunhofer.de)
# Fabian Bergmann (fabian.bergmann@fokus.fraunhofer.de)
# Henri Willems (henri.willems@fokus.fraunhofer.de)
# Updated: Amir Arfan 2021, MATH275 Project

import torch
from torch import nn

import torchvision

import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.color_palette("bright")


class QonvLayer(nn.Module):
    def __init__(
        self,
        stride=2,
        device="default.qubit",
        wires=4,
        circuit_layers=4,
        n_rotations=8,
        out_channels=4,
        seed=None,
    ):
        super(QonvLayer, self).__init__()

        # init device
        self.wires = wires
        self.dev = qml.device(device, wires=self.wires)

        self.stride = stride
        self.out_channels = min(out_channels, wires)

        if seed is None:
            seed = np.random.randint(low=0, high=10e6)

        print("Initializing Circuit with random seed", seed)

        # random circuits
        @qml.qnode(device=self.dev, interface="torch")
        def circuit(inputs, weights):
            n_inputs = 4
            # Encoding of 4 classical input values
            for j in range(n_inputs):
                qml.RY(inputs[j], wires=j)
            # Random quantum circuit
            RandomLayers(weights, wires=list(range(self.wires)), seed=seed)

            # Measurement producing 4 classical output values
            return [
                qml.expval(qml.PauliZ(j)) for j in range(self.out_channels)
            ]

        weight_shapes = {"weights": [circuit_layers, n_rotations]}
        self.circuit = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes)

    def draw(self):
        # build circuit by sending dummy data through it
        _ = self.circuit(inputs=torch.from_numpy(np.zeros(4)))
        print(self.circuit.qnode.draw())
        self.circuit.zero_grad()

    def forward(self, img):
        bs, h, w, ch = img.size()
        if ch > 1:
            img = img.mean(axis=-1).reshape(bs, h, w, 1)

        kernel_size = 2
        h_out = (h - kernel_size) // self.stride + 1
        w_out = (w - kernel_size) // self.stride + 1

        out = torch.zeros((bs, h_out, w_out, self.out_channels))

        # Loop over the coordinates of the top-left pixel of 2X2 squares
        for b in range(bs):
            for j in range(0, h_out, self.stride):
                for k in range(0, w_out, self.stride):
                    # Process a squared 2x2 region of the image with a quantum circuit
                    q_results = self.circuit(
                        inputs=torch.Tensor(
                            [
                                img[b, j, k, 0],
                                img[b, j, k + 1, 0],
                                img[b, j + 1, k, 0],
                                img[b, j + 1, k + 1, 0],
                            ]
                        )
                    )
                    # Assign expectation values to different channels of the output pixel (j/2, k/2)
                    for c in range(self.out_channels):
                        out[
                            b, j // kernel_size, k // kernel_size, c
                        ] = q_results[c]

        return out


def transform(x):
    x = np.array(x)
    x = x / 255.0

    return torch.from_numpy(x).float()


def train(model, train_loader, max_i=500, epochs=50):
    print("Starting Training for {} epochs".format(epochs))

    model.train()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    losses = np.array([])
    accs = np.array([])

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):

            # prepare inputs and labels
            x = x.view(-1, 28, 28, 1)
            y = y.long()

            # reset optimizer
            optimizer.zero_grad()

            # engage
            y_pred = model(x)

            # error, gradients and optimization
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            # output
            acc = accuracy_score(y, y_pred.argmax(-1).numpy())

            accs = np.append(accs, acc)
            losses = np.append(losses, loss.item())

            print(
                "Epoch:",
                epoch,
                "\tStep:",
                i,
                "\tAcc:",
                round(acc, 3),
                "\tLoss:",
                round(loss.item(), 3),
                "\tMean Loss:",
                round(float(losses[-30:].mean()), 3),
                "\tMean Acc:",
                round(float(accs[-30:].mean()), 3),
            )
            print("\nGradients Layer 0:")
            print(model[0].circuit.weights.grad)

            if i % 5 == 0:
                model[0].draw()

            print("---------------------------------------\n")

            if i >= max_i:
                break

    return model, losses, accs


if __name__ == "__main__":

    please_train = True

    # prepare dataset

    # build the model

    if please_train:
        train_set = torchvision.datasets.MNIST(
            root="./mnist", train=True, download=True, transform=transform
        )
        test_set = torchvision.datasets.MNIST(
            root="./mnist", train=False, download=True, transform=transform
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=4, shuffle=True
        )

        model = torch.nn.Sequential(
            QonvLayer(
                stride=2,
                circuit_layers=2,
                n_rotations=4,
                out_channels=4,
                seed=9321727,
            ),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=14 * 14 * 4, out_features=10),
        )
        model, losses, accs = train(model, train_loader, max_i=300, epochs=1)
        loss = np.array(losses)
        accs = np.array(accs)
        np.save("accs.npy", accs)
        np.save("loss.npy", loss)
    else:
        accs = np.load("accs.npy")
        loss = np.load("loss.npy")

    # plot losses and accuracies
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 9))
    # ax1.plot(losses)
    # ax1.set_title("Loss")
    # ax1.set_xlabel("Steps")
    # ax1.set_ylabel("Loss")
    sns.lineplot(data=loss, ax=ax2)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Loss")
    plt.savefig("mnist_trainableqonv_loss.png")
    plt.show()

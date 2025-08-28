# PiGLeT: Probabilistic Message Passing for Semi-supervised Link Sign Prediction (ICDM'25)

This repository is the official implementation of **PiGLeT: Probabilistic Message Passing for Semi-supervised Link Sign Prediction (ICDM'25)**, a probabilistic message-passing model designed for semi-supervised link sign prediction tasks.

## Abstract📌
How can we accurately predict the signs of unseen links in partially observed signed graphs? Signed graphs are widely used to represent complex relationships in areas such as social and biological networks. Although prior methods enhance representation learning by extending Graph Neural Networks
with social theories, they rely on the unrealistic assumption that all link signs are known. In practice, however, link signs are often only partially labeled due to the high cost or difficulty of obtaining ground-truth annotations. For example, in the Bitcoin transaction network, while some interactions between users can be labeled as trusted or untrusted based on external investigations or contextual evidence, many other interactions lack explicit labels.

In this work, we propose PiGLeT (PROBABILISTIC MESSAGE PASSING FOR SEMI-SUPERVISED LINK SIGN PREDICTION), a novel approach for accurate link sign prediction on signed graphs with partially observed sign labels. The main idea is to probabilistically interpret unlabeled links as both positive and
negative based on a soft-labeling strategy, allowing information to be propagated through both types of edges. To reduce the impact of edges with uncertain and potentially noisy soft labels, we additionally propose confidence-based weights on unlabeled edges during message passing. PiGLeT adaptively balances the importance of edges regardless of label availability by employing relation-aware attention scores as well. Since propagating messages with accurate soft labels for unlabeled links leads to improved node embeddings, PiGLeT iteratively refines both node embeddings and soft labels during training, establishing a theoretical connection to the Expectation-Maximization (EM) algorithm. Extensive experiments on five real-world datasets
under a semi-supervised setting show that PIGLET consistently outperforms existing methods, achieving up to 7.6% improvement in AUC and 11.0% in Macro-F1.

## Requirements 📋

Ensure you have Python installed (recommended Python 3.8+). The required packages are:

```
matplotlib==3.7.5
numpy==1.24.3
scikit_learn==1.3.2
scipy==1.15.3
seaborn==0.13.2
torch==1.13.1
torch_geometric==2.6.1
torch_geometric_signed_directed==0.25.0
torcheval==0.0.7
```

You can install all required dependencies using:

```bash
pip install -r requirements.txt
```

## Installation ⚙️

Clone this repository to your local machine:

```bash
git clone https://github.com/piglet-anonymous/PiGLeT.git
cd PiGLeT
```

## Usage 💻

To train and evaluate the PiGLeT model, use:

```bash
python main.py --epochs 100 --dataset slashdot
```

Modify the arguments in the command line to change datasets, epochs, and other experiment settings.

## Code Structure 📂

* `main.py`: Entry point for running experiments and evaluations.
* `src/train.py`: Contains the training loop logic and performance evaluations.
* `src/utils.py`: Common utility functions.
* `src/model/Piglet.py`: Defines the PiGLeT model architecture.
* `src/model/PigletConv.py`: Implements the probabilistic message-passing convolutional layers.



## Citation

Please cite the following paper if you use our code:
```
@inproceedings{park2025piglet,
  title={PiGLeT: Probabilistic Message Passing for Semi-supervised Link Sign Prediction},
  author={Park, Ka Hyun and Kim, Junghun and Jung, Jinhong and Kang, U},
  booktitle={2024 IEEE International Conference on Data Mining (ICDM)},
  year={2025},
  organization={IEEE}
}
```

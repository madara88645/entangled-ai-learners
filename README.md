[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/madara88645/entangled-ai-learners/blob/main/Entangled-AI/entangled_models_final.ipynb)

# Entangled AI Learners (CNN + MLP)

This project demonstrates an experimental framework for entangled learning between two heterogeneous models (a Convolutional Neural Network and a Multi-Layer Perceptron) using the MNIST dataset.

## Overview

- **Models**: CNN and MLP
- **Dataset**: MNIST
- **Loss Function**: Categorical Crossentropy + KL Divergence (Entangled Loss)
- **Entanglement Strength (λ)**: Increases dynamically over epochs

## Files

- `entangled_models_final.ipynb` - Full Jupyter notebook with training and results
- `entangled_utils.py` - Helper functions for entangled loss and lambda scheduler
- `requirements.txt` - Dependencies

## Results

- **CNN Accuracy**: ~99.6%
- **MLP Accuracy**: ~98.7%
- Demonstrates effective information transfer via entangled output feedback

## License

MIT



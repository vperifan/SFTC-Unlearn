## SFTC: Machine Unlearning via Selective Fine-tuning and Targeted Confusion

This repository provides the implementation of SFTC: Machine Unlearning via Selective Fine-tuning and Targeted Confusion. The [paper](https://dl.acm.org/doi/10.1145/3655693.3655697) received the best paper award in European Interdisciplinary Cybersecurity Conference (EICC 2024).

### Summary
In SFTC, we design an algorithm for unlearning based on teacher-student approaches. Specifically, we draw inspiration from the [Bad Teaching](https://ojs.aaai.org/index.php/AAAI/article/view/25879) and [SCRUB](https://arxiv.org/pdf/2302.09880.pdf) methods.

Bad Teaching achieves unlearning by minimizing the KL divergence on the retain set using predictions from the original model, while simultaneously minimizing the KL divergence on the forget set by following a random output distribution.

SCRUB performs unlearning by fine-tuning on the retain set, minimizing the KL divergence on the retain set using predictions from the original model and maximizing the KL divergence on the forget set using predictions from the original model.

Both approaches try to confuse the original model on the forget set (Bad Teaching by using random outputs and SCRUB by maximizing the KL divergence). However, these methods lack control over the influence on samples in the retain set that are similar to those in the forget set. This lack of control can lead to unintended confusion on more samples than desired.

To address this, we developed SFTC, which allows us to control the propagated confusion. We initialize a random generator that, based on a confusion fraction, biases its outputs towards the correct class for each sample in the forget set. In practice, this biased generator produces random outputs that are correct concerning the class sample, aiming to lower the original model's confidence on the forget set and prevent error propagation to nearby samples.

During optimization, SFTC fine-tunes the model on the retain set (similar to SCRUB), minimizes the KL divergence on the retain set using predictions from the original model (similar to both Bad Teaching and SCRUB) and minimizes the KL divergence on the forget set using our biased random generator.

### Unlearning
* Refer to the [example notebook](Example.ipynb) for unlearning.

### Datasets 
* CIFAR-10
  * For CIFAR-10, we use the forget set introduced on the [Starting Kit for the NeurIPS 2023 Machine Unlearning Challenge](https://github.com/unlearning-challenge/starting-kit)
* [MUFAC](https://github.com/ndb796/MachineUnlearning/tree/main)
  * For MUFAC, we use the forget set introduced in the corresponding [repo](https://github.com/ndb796/MachineUnlearning/tree/main) of the [Towards Machine Unlearning Benchmarks: Forgetting the Personal Identities in Facial Recognition Systems](https://arxiv.org/abs/2311.02240) paper.
* [FER](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
  * It is a forget set curated by our team, where the goal is to perform in-context unlearning on images corresponding to minors labeled with sadness or fear.

### Unlearning Algorithms

- [X] Unlearning by Fine-tuning on the retain set
- [X] [EU-k Forgetting](https://arxiv.org/abs/2201.06640)
- [X] [CF-k Forgetting](https://arxiv.org/abs/2201.06640)
- [X] [NegGrad](https://arxiv.org/abs/1911.04933)
- [X] [Advanced NegGrad](https://arxiv.org/pdf/2311.02240.pdf)
- [X] [Bad Teaching](https://ojs.aaai.org/index.php/AAAI/article/view/25879)
- [X] [SCRUB](https://arxiv.org/pdf/2302.09880.pdf)
- [X] [SFTC](https://dl.acm.org/doi/10.1145/3655693.3655697)

### Citation
```
@inproceedings{10.1145/3655693.3655697,
    author = {Perifanis, Vasileios and Karypidis, Efstathios and Komodakis, Nikos and Efraimidis, Pavlos},
    title = {SFTC: Machine Unlearning via Selective Fine-tuning and Targeted Confusion},
    year = {2024},
    isbn = {9798400716515},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3655693.3655697},
    doi = {10.1145/3655693.3655697},
    booktitle = {European Interdisciplinary Cybersecurity Conference},
    pages = {29–36},
    numpages = {8},
    keywords = {Data Privacy, Deep Learning, Machine Learning Security and Privacy, Machine Unlearning},
    location = { Xanthi, Greece, },
    series = {EICC 2024}
}
```

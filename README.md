## Disentangled Multi-Fidelity Deep Bayesian Active Learning
## Paper: 
Dongxia Wu, Ruijia Niu, Matteo Chinazzi, Yi-An Ma, Rose Yu, [Disentangled Multi-Fidelity Deep Bayesian Active Learning](https://arxiv.org/pdf/2305.04392.pdf), ICML 2023

## Requirements

Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```


## Abstract
To balance quality and cost, various domain areas of science and engineering run simulations at multiple levels of sophistication. Multi-fidelity active learning aims to learn a direct mapping from input parameters to simulation outputs by actively acquiring data from multiple fidelity levels. However, existing approaches based on Gaussian processes are hardly scalable to high-dimensional data. Other deep learning-based methods use the hierarchical structure, which only supports passing information from low-fidelity to high-fidelity. These approaches can lead to the undesirable propagation of errors from low-fidelity representations to high-fidelity ones. We propose a novel disentangled deep Bayesian learning framework for multi-fidelity active learning, that learns the surrogate models conditioned on the distribution of functions at multiple fidelities. 


## Description
1. dmfdal_2f/: Experiments on tasks with 2 fidelity levels, including heat2, poisson2, and fliud.
2. dmfdal_3f/: Experiments on tasks with 3 fidelity levels, including heat3 and poisson3.

## Dataset Download

### D-MFDAL 2F
```
cd dmfdal_2f/
wget -O data.zip https://roselab1.ucsd.edu/seafile/f/b40ebc08daac417eb5b3/?dl=1
unzip data.zip
```

### D-MFDAL 3F
```
cd dmfdal_3f/
wget -O data.zip https://roselab1.ucsd.edu/seafile/f/1bbdc8ab0791471cbf26/?dl=1
unzip data.zip
```

## Model Training and Evaluation

### D-MFDAL 2F
```
cd dmfdal_2f/
./heat.sh
./poisson.sh
./fluid.sh
```

### D-MFDAL 3F
```
cd dmfdal_3f/
./heat3.sh
./poisson3.sh
```


## Cite
```
@article{wu2023disentangled,
  title={Disentangled Multi-Fidelity Deep Bayesian Active Learning},
  author={Wu, Dongxia and Niu, Ruijia and Chinazzi, Matteo and Ma, Yian and Yu, Rose},
  booktitle={International Conference on Machine Learning},
  organization={PMLR},
  year={2023}
}
```

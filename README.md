# Transferability Metric Toolkit

This repository provides an implementation of various transferability metrics for evaluating the performance of pre-trained models in transfer learning tasks. The supported methods include:

- [An Information-theoretic Approach to Transferability in Task Transfer Learning (H-Score, ICIP 2019)](http://yangli-feasibility.com/home/media/icip-19.pdf)
- [LEEP: A New Measure to Evaluate Transferability of Learned Representations (LEEP, ICML 2020)](http://proceedings.mlr.press/v119/nguyen20b/nguyen20b.pdf)
- [Log Maximum Evidence in `LogME: Practical Assessment of Pre-trained Models for Transfer Learning (LogME, ICML 2021)](https://arxiv.org/pdf/2102.11005.pdf)
- [Negative Conditional Entropy in `Transferability and Hardness of Supervised Classification Tasks (NCE, ICCV 2019)](https://arxiv.org/pdf/1908.08142v1.pdf)
- [OTCE: A Transferability Metric for Cross-Domain Cross-Task Representations （OTCE, CVPR2021）](https://arxiv.org/abs/2103.13843)

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

Please refer to the [main.py](main.py) file for a complete example of how to use the transferability metrics. The general process is as follows:

1. Import the necessary functions from the `tool.metric` module.
2. Define a list of directory sets, where each set contains the target root directory, target prediction directory, and source root directory.
3. Initialize a dictionary to hold the results, and iterate over each directory set to calculate the metrics.
4. Print the results.

The `tool.metric` module provides the following functions that accept specific input directories:

- `h_score(tar_root_dir)`
- `log_expected_empirical_prediction(tar_predictz_dir)`
- `log_maximum_evidence(tar_root_dir)`
- `negative_conditional_entropy(src_root_dir, tar_predictz_dir)`
- `optimal_transport(src_root_dir, tar_root_dir)`

## Example Results
Source domain:amazon
Model:Resnet

| Domain   | Finetuned Acc | HScore | LEEP     | LogME | NCE | OTCE |
| -------- | --------      | ------ |-------   | -----| --- | ----- |
| dslr     |    95.95      | 335.26 |   -0.95  | 0.20  |  -0.53 | -0.83|
| webcam   |  72.83        | -12.07 |   -1.80  | 0.54  |  -2.30 | -1.64|

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

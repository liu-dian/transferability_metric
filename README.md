# transferability_metric
## Supported Methods

Supported methods include:

- [An Information-theoretic Approach to Transferability in Task Transfer Learning (H-Score, ICIP 2019)](http://yangli-feasibility.com/home/media/icip-19.pdf)

- [LEEP: A New Measure to Evaluate Transferability of Learned Representations (LEEP, ICML 2020)](http://proceedings.mlr.press/v119/nguyen20b/nguyen20b.pdf)

- [Log Maximum Evidence in `LogME: Practical Assessment of Pre-trained Models for Transfer Learning (LogME, ICML 2021)](https://arxiv.org/pdf/2102.11005.pdf)

- [Negative Conditional Entropy in `Transferability and Hardness of Supervised Classification Tasks (NCE, ICCV 2019)](https://arxiv.org/pdf/1908.08142v1.pdf)

- [OTCE: A Transferability Metric for Cross-Domain Cross-Task Representations](https://arxiv.org/abs/2103.13843)
  
| Domain   | Finetuned Acc | HScore | LEEP     | LogME | NCE | OTCE |
| -------- | --------      | ------ |-------   | -----| --- | ----- |
| dslr     |    95.95      | 335.26 |   -0.95  | 0.20  |  -0.53 | -0.83|
| webcam   |  72.83        | -12.07 |   -1.80  | 0.54  |  -2.30 | -1.64|

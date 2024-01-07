# Background

In order to make more people learn how to use this toolkit better, we introduce the background of the research field of the transfer learning, especially about the transferability metrics.

Transfer learning aims to re-use knowledge learned on a source task to help learning a target task, for which typically there is scarce training data. Moreover, training models from scratch can be excessively costly and time-consuming. The most prevalent method of transfer learning is to pre-train a source model on a large source dataset, and then fine-tune it on the target dataset. However, different target tasks benefit from using different source model architectures or pre-training on different source datasets. Hence, a key challenge is determining which source model is best suited for which target task, and doing so in a computationally efficient manner.

An imperative question in transfer learning is transferability, i.e. when a transfer may work and to what extent. Traditionally, transferability is measured purely empirically using model loss or accuracy on the validation set. Transferability metrics provide heuristics for selecting the most suitable source models for a given target dataset, by avoiding directly transferring all the models. These methods generally work by applying a source model to the target dataset to compute embeddings or predictions. Then they efficiently assess how compatible these embeddings/predictions are with the target labels. This provides a proxy for how well the source model transfers to the target task.

## H-score
H-score is based on the intuition that a model transfers well to a target dataset if the target embeddings have low inter-class variance and low feature redundancy. These quantities are computed by constructing the interclass and data covariance matrices. 
## LEEP  
LEEP begins by leveraging the source model to generate pseudo-labels for the target dataset. Subsequently, it evaluates the empirical conditional distribution, which quantifies the probability of the actual target labels given these pseudo-labels from the source model. These probabilities are then utilized to calculate the log-likelihood, drawing a comparison between the actual target labels and the predictions made by the source model. The underlying principle is that if the source model's predictions are concentrated around the true target labels—forming distinct clusters, so to speak—then the model's adaptation to the target dataset is likely to be more successful.
## LogME 
After embedding the target images using the source feature extractor, LogME computes the probability of the target labels conditioned on these embeddings (i.e. the evidence of target labels). By setting up a graphical model and using independence assumptions between samples, the authors propose an efficient algorithm for computing such evidence.
## NCE
NCE adopts conditional entropy to evaluate transferability and task hardness under a particular setting, i.e., source and target tasks share the same input instances but different labels. They provide a derivation that the empirical transferability is lower bounded by the negative conditional entropy.
## OTCE

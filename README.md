# Bayesian CNNs and DNA
 Temporariry Github repository for the DNA project

# To do
 - [] Work out template code for us to build on
  - [] Implement final fully connected linear layer
  - [] Output now is a 2d vector, labels are scalar, need to convert using dataloader
 - [] Figure out a solution for the different sequence lengths
  - [] possible: gene2vec
  - [] flexible/dynamic k-max pooling
 - [] Implement dataloader (can be one done once we know exact formats per person per gene)
 - [] Implement actual training of the CNN
 - - [] Need to fix the output label of the network scalar --> vector (or visa versa)

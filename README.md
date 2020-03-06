# Bayesian CNNs and DNA
 Temporariry Github repository for the DNA project

# How the current files work
Within the file "code" it is possible to run the following files:
- Main.py: possible to forward one (fake) created batch through the CNN as created in the same file
- create_arbitrary_data.py: Creates arbritrary BOW vectors, appends them in a matrix to create a sequence, which can be used to create fake data as a holder until we have access to the real data
- WGS_to_embeds.py: possible to go from one sequence "AGTC" to a tensor, also possible to add new sequences to the same batch.
- CNN.py: some template code for a 3d image (RGB) can be used to see how a previous project succeeded
- train_cnn.py: some template code for how to train the network


# To do
 - [ ] Work out template code for us to build on
  - - [x] Implement CNN as proposed in the document
  - - [ ] Implement final fully connected linear layer
  - - [ ] Output now is a 2d vector, labels are scalar, need to convert using dataloader
 - [ ] Figure out a solution for the different sequence lengths
  - - [ ] possible: gene2vec
  - - [ ] flexible/dynamic k-max pooling
 - [ ] Implement dataloader (can be one done once we know exact formats per person per gene)
 - [ ] Implement actual training of the CNN
 - - [ ] Need to fix the output label of the network scalar --> vector (or visa versa)

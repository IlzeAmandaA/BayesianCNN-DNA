import torch
import torch.nn as nn
import torch.nn.functional as F



class AttentionNetwork(nn.Module):

    def __init__(self):
        super(AttentionNetwork, self).__init__()

        #define the dimensions of the FFNN
        self.L=500
        self.D=128
        self.K=1 #final output dimension


        #transform the data using a CNN
        self.transformer_part1 =nn.Sequential(
            #here pass the CNN settings
            # nn.Conv1d()
            # .. .
            # ...

        )

        #
        self.transformer_part2 = nn.Sequential(
            nn.Linear('mattch with output of part1', self.L),
            nn.ReLU()
        )


        #estiamte the attention weights
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        #the actual classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        #check input dimensions
        print('Input dim', x.shape)

        #tranform the input if needed
        #x=x.squeeze(0)

        H = self.transformer_part1(x)
        #check output dimensions
        print('H dim', H.shape)
        #tranform if necessary H.view(-1, 50*4*4)
        H = self.transformer_part2(H)

        A = self.attention(H)
        #check output dimensions of A
        print('A dim', A.shape)
        #transform if needed A = torch.transpose(A, 1,0)
        A=F.softmax(A, dim=1) #over the second dimension as one to extract value for each feature

        #apply attention to the created embeddings
        M = torch.mm(A,H)

        #pass the created embedding to the classifier
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float() #transform to 0 or 1  float

        return Y_prob, Y_hat, A

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]

        return error, Y_hat






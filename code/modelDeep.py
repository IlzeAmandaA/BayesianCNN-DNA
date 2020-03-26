import torch
import torch.nn as nn
import torch.nn.functional as F



class AttentionNetworkDeep(nn.Module):

    def __init__(self, L,D,k_max,hidden_CNN):
        super(AttentionNetworkDeep, self).__init__()

        #define the dimensions of the FFNN
        self.L = L  # 1000
        self.D = D  # 128
        self.K = 1  # final output dimension
        self.k_max = k_max  # 20
        self.hidden_CNN = hidden_CNN  # 128
        self.dim = 2


        #transform the data using a CNN
        self.transformer_part1 =nn.Sequential(
            nn.Conv1d(4,self.hidden_CNN,kernel_size=11, stride=1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(self.hidden_CNN,self.hidden_CNN, kernel_size=11),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

        self.transformer_part2 = nn.Sequential(
            nn.Linear(self.k_max*self.hidden_CNN, self.L), #match the first numer with the output of cnn after pooling 20*
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)


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
        #assumes input of form gene_count x channels x sequnce_length
        #example: 20x4x2000
        H = self.transformer_part1(x)

        #applying pooling here
        H = self.kmax_pooling(H) #genes x kernels x max_feat

        #flatten out
        H = H.reshape(-1, self.hidden_CNN*self.k_max)
        #pass to the linear tranforamtion
        H = self.transformer_part2(H) #N_genes x L

        A_V = self.attention_V(H)
        A_U = self.attention_U(H)

        A = self.attention_weights(A_V * A_U) #element wise multiplication

     #   A = self.attention(H)
        A = torch.transpose(A, 1,0)
        A=F.softmax(A, dim=1)

        M = torch.mm(A,H)

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    def kmax_pooling(self, x):
        index = x.topk(self.k_max, dim = self.dim)[1].sort(dim = self.dim)[0]
        return x.gather(self.dim, index)

    def calculate_objective(self, X, Y):
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.-1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
        return neg_log_likelihood, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item() #.data[0]
        return error, Y_hat






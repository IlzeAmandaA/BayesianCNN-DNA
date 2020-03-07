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
            nn.Conv1d(1,10,kernel_size=11, stride=1, padding=1),  #second number specifies output chanels
            # here implement a dropout
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2), #the paper has globablmax pooling
            # #here implement a dropout
            nn.Conv1d(20,50, kernel_size=11, stride=1),
            #implementd dropput
            nn.ReLU()
        ) #hence here the output will be [1, 10, z] where z will vary depending on the input gene length

        #
        self.transformer_part2 = nn.Sequential(
            nn.Linear('matched with output of the cnn', self.L),
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

    def kmax_pooling(self, x, dim, k):
        index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
        return x.gather(dim, index)

    def forward(self, x):
        #check input dimensions
        print('Input dim', x.shape) #pass the first bag (person)

        ### still would have to pass gene by gene
        # due to difference in len (cannot store in a matrix)
        stack_genes=[]
        for gene in x:
            H = self.transformer_part1(gene)
            H = self.kmax_pooling(H,dim=2, k=20)
            print('dim after kmax pooling ', H.shape)
            stack_genes.append(H)

        stack_genes = torch.stack(stack_genes)
            #check output dimensions

        print('all genes dim', stack_genes.shape)
        #need to reshape here
        stack_genes = stack_genes.view(-1, stack_genes.shape[1]*stack_genes.shape[2])
        #result should be genes x embedding
        print('Tranfored stack genes:', stack_genes.shape)

        H = self.transformer_part2(stack_genes)



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
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1.-1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]

        return error, Y_hat






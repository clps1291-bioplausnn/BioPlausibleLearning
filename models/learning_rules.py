import torch
import torch.nn as nn


class HebbLayer(nn.Module):
     def __init__(self, input_dim, output_dim, lr, require_hebb=True, activation=True, update_rule='hebb',p=None):
        super(HebbLayer, self).__init__()
        # Init weights
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim) * 0.01, requires_grad=False) # ban bp for hebb layers
        self.require_hebb = require_hebb
        self.past_product_sum = torch.zeros_like(self.weights.data)
        self.stimuli_times =  0
        self.relu = nn.ReLU()
        self.activation = activation
        self.lr = lr
        self.update_rule = update_rule
        self.p = p

        
     def forward(self, x): # Forward call triggers the update!
        z = self.get_product(x)
        # perform update during forward pass if required
        if self.require_hebb:
            # Select update rule
            if self.update_rule == 'hebb':
                self.hebb_update(x)  
            elif self.update_rule == 'oja':
                self.oja_update(x) 
            elif self.update_rule == 'gupta':
                if self.p is None:
                    raise ValueError("Percentile 'p' must be provided for Gupta update rule")
                self.gupta_update(x)  
            elif self.update_rule == 'modified_gupta':
                if self.p is None:
                    raise ValueError("Percentile 'p' must be provided for modified Gupta update rule")
                self.modified_gupta_update(x)  
            else:
                raise ValueError("Invalid update rule specified")
        return z
     
     # 1.naive hebbian learning rule
     def hebb_update(self,x):
         z = self.get_product(x) # (bs,50)
         self.weights.data += self.lr * torch.matmul(x.t(),z) # (bs,784).T x (bs,50) --> (784,50) 
         ## Consider torch.einsum for tensor/matrix multiplication ##

     # 2.Oja's rule
     def oja_update(self,x):
         z = self.get_product(x) # (bs,50)
         self.weights.data += self.lr * torch.matmul((x.t()-torch.matmul(self.weights, z.t())), z) # [(bs,784).T - (784,50) x (bs,50).T] x (bs,50)--> (784,50)

     # 3.modified hebbian learning rule with thresholding and gradient sparsity by Gupta et al
     def gupta_update(self,x):
         z = self.get_product(x) # (bs,50)
         self.stimuli_times +=1
         delta_w = self.lr * torch.matmul(x.t(), z) 
         # thresholding
         if self.stimuli_times>1:
             delta_w -= self.past_product_sum /(self.stimuli_times-1)
         # gradient sparsity
         threshold = self.find_percentile(delta_w)
         mask = delta_w >= threshold
         delta_w *= mask
         self.weights.data += delta_w
         self.record_stimuli(product=torch.matmul(x.t(), z))
     
     # 4.modified Gupta's rule with local gradient sparsity instead of global gradient sparsity
     def modified_gupta_update(self,x):
         pass

     def record_stimuli(self,product):
         self.past_product_sum += product

     def get_product(self,x):
         z = torch.matmul(x, self.weights) # (bs,784) x (784,2000) = (bs,2000)
         if self.activation:
             return self.relu(z)
         return z
     
     def find_percentile(self,tensor):
        """
        Find the p-th percentile value in a tensor.
        """
        k = 1 + round(.01 * float(self.p) * (tensor.numel() - 1))
        return tensor.view(-1).kthvalue(k).values.item()
         
#TODO
#krotov
def get_unsupervised_weights(X, n_hidden, n_epochs, batch_size, 
        learning_rate=2e-2, precision=1e-30, anti_hebbian_learning_strength=0.4, lebesgue_norm=2.0, rank=2):
    sample_sz = X.shape[1]
    weights = torch.rand((n_hidden, sample_sz), dtype=torch.float, device='cuda')
    for epoch in range(n_epochs):    
        eps = learning_rate * (1 - epoch / n_epochs)
        shuffled_epoch_data = X[torch.randperm(X.shape[0]),:]
        for i in range(X.shape[0] // batch_size):
            mini_batch = shuffled_epoch_data[i*batch_size:(i+1)*batch_size,:].cuda()            
            mini_batch = torch.transpose(mini_batch, 0, 1)            
            sign = torch.sign(weights)            
            W = sign * torch.abs(weights) ** (lebesgue_norm - 1)        
            tot_input=torch.mm(W, mini_batch)            
            
            y = torch.argsort(tot_input, dim=0)            
            yl = torch.zeros((n_hidden, batch_size), dtype = torch.float).cuda()
            yl[y[n_hidden-1,:], torch.arange(batch_size)] = 1.0
            yl[y[n_hidden-rank], torch.arange(batch_size)] =- anti_hebbian_learning_strength            
                    
            xx = torch.sum(yl * tot_input,1)            
            xx = xx.unsqueeze(1)                    
            xx = xx.repeat(1, sample_sz)                            
            ds = torch.mm(yl, torch.transpose(mini_batch, 0, 1)) - xx * weights            
            
            nc = torch.max(torch.abs(ds))            
            if nc < precision: nc = precision            
            weights += eps*(ds/nc)
    return weights
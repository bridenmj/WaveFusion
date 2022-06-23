import torch
from torch import nn
import torch.nn.functional as F


################################### WLCNN MODEL ###################################

class WLCNN(nn.Module):
    """
    WLCNN convolves over the 61x32x250 scalograms. 
    """
    def __init__(self):
        super(WLCNN,self).__init__()

        self.conv1a = nn.Conv2d(61,122, kernel_size=(4,4), stride=(1,3)) 
        self.a1 = nn.ReLU()
        self.conv1a_bn = nn.BatchNorm2d(122)


        self.conv1b = nn.Conv2d(122,122, kernel_size=(4,2), stride=(1,3)) ####NEED TO CHANGE KERNEL TO 4,3
        self.a2 = nn.ReLU()
        self.conv1b_bn = nn.BatchNorm2d(122)
        
        self.maxPool1 = nn.MaxPool2d((2,2))

        self.conv2a = nn.Conv2d(122,244, kernel_size=(3,3), stride=(2,2))
        self.a3 = nn.ReLU()
        self.conv2a_bn = nn.BatchNorm2d(244)

        self.conv2b = nn.Conv2d(244,244, kernel_size=(4,4), stride=(2,2))
        self.a4 = nn.ReLU()
        self.conv2b_bn = nn.BatchNorm2d(244)

        self.maxPool2=nn.MaxPool2d((2,2))
        
        self.fc1 = nn.Linear(244,3) 



    def forward(self,x):
 
        x = self.conv1a(x)
        x = self.a1(x)
        x = self.conv1a_bn(x)

      
        x = self.conv1b(x)
        x = self.a2(x)
        x = self.conv1b_bn(x)
        
        x = self.maxPool1(x)

  
        x = self.conv2a(x)
        x = self.a3(x)
        x = self.conv2a_bn(x)


        x = self.conv2b(x)
        x = self.a4(x)
        x = self.conv2b_bn(x)

        x = self.maxPool2(x)
        
        x = x.view(-1,244)

        x = self.fc1(x) #a (1x1) output for each lead
       # print(x.shape)

        
        return x
    
################################### W3DCNN MODEL ###################################   
 
class W3DCNN(nn.Module):
    """
    W3DCNN convolves over the 61x32x250 scalograms. 
    """
    def __init__(self):
        super(W3DCNN,self).__init__()

        self.conv1a = nn.Conv3d(1,4, kernel_size=(3,4,4), stride=(2,1,3)) 
        self.a1 = nn.ReLU()
        self.conv1a_bn = nn.BatchNorm3d(4)


        self.conv1b = nn.Conv3d(4,8, kernel_size=(4,4,2), stride=(2,1,3))
        self.a2 = nn.ReLU()
        self.conv1b_bn = nn.BatchNorm3d(8)
        
        self.maxPool1 = nn.MaxPool3d((2,2,2))

        self.conv2a = nn.Conv3d(8,16, kernel_size=(3,3,4), stride=(2,2,2))
        self.a3 = nn.ReLU()
        self.conv2a_bn = nn.BatchNorm3d(16)

        self.conv2b = nn.Conv3d(16,32, kernel_size=(2,4,4), stride=(1,2,2))
        self.a4 = nn.ReLU()
        self.conv2b_bn = nn.BatchNorm3d(32)

        self.maxPool2=nn.MaxPool3d((2,2,2))
        
        self.fc1 = nn.Linear(32,3) 



    def forward(self,x):

        bs = len(x[:,0,0,0])
        chans = len(x[0,:,0,0])
        x = x.view(bs,1,chans,32,250)

        x = self.conv1a(x)
        x = self.a1(x)
        x = self.conv1a_bn(x)

      
        x = self.conv1b(x)
        x = self.a2(x)
        x = self.conv1b_bn(x)
        
        x = self.maxPool1(x)

        x = self.conv2a(x)
        x = self.a3(x)
        x = self.conv2a_bn(x)


        x = self.conv2b(x)
        x = self.a4(x)
        x = self.conv2b_bn(x)

        x = self.maxPool2(x)
        
        x = x.view(-1,32)

        x = self.fc1(x) #a (1x1) output for each lead

        
        return x
    

    
################################### WAVEFUSION MODEL ###################################
    
class Wave_Fusion_Model(nn.Module):
    """
    Wave Fusion Model. Contains 64 Wave Lead Convs that convolves over each 
    eeg Lead. Implements multiplicative fusion loss by Liu
    et al https://arxiv.org/pdf/1805.11730.pdf
    """
    def __init__(self, beta):
        self.leads = 61
        self.beta = beta
        super(Wave_Fusion_Model,self).__init__()
        for i in range(self.leads):
            self.add_module('Wave_Lead_Conv' + str(i), Wave_Lead_Conv())
        self.Wave_Lead_Conv = AttrProxy(self, 'Wave_Lead_Conv')



    def forward(self, x):
        """
        feeds each eeg channel in x to a Wave_Lead_Conv
        x: a tensor of shape BatchSize x self.leads x 32 x 256
        returns: 
        On training: a list of size (batchsize, num_lead) w/ each entry a [1,2] tensor of softmax probabilities for each class
        On eval: a tensor containing the class losses for each data in the batch
        """
        tmp = []
        preds = []
        bs = len(x[:,0,0,0])

        #feed data to the Wave_Lead_Convs and return list of predictions
        for i in range(bs):
            tmp = []
            for j in range(self.leads):
                #each lead to a wave_lead_conv. reshape to 1,1,32,250
                #print(x[i,j,:,:].shape)
                t1 = self.Wave_Lead_Conv.__getitem__(j)(x[i,j,:,:].view(1,1,32,250))
                t1 = t1.clone()
                #t1.retain_grad()
                tmp.append(t1)
            
            preds.append(tmp)
        #training prediction is a list of size (batchsize, num_lead) w/ each entry a [1,2] tensor of softmax probabilities for each class
        if self.training:
            return preds
        #eval predictions are the loss for each class. 
        else:

            #Liu Method
            pred_tmp = []
            for i in range(bs):
                loss0 = 0
                loss1 = 0
                loss2 = 0
                
                #compute the class loss0
                for j in range(self.leads):
                    #if preds[i][j][0] != 1 and preds[i][j][0] != 0: 
                    #compute weighting coefficient
                    coeff = 1
                    for k in range(self.leads):

                        if k != j:
                            coeff = coeff * (1-preds[i][k][0])

                    #detach coefficient from computation graph
                    coeff=coeff.detach()

                    #calculate loss, add
                    loss0 += (-1.0)*torch.pow( coeff, self.beta/(self.leads-1) )*torch.log(preds[i][j][0])
                    
                    #loss0 += (-1.0)*torch.log(preds[i][j][0])
                
                #calculate class loss 1
                for j in range(self.leads):
                    
                    #if preds[i][j][1] != 1 and preds[i][j][1] != 0:
                    #compute weighting coefficient
                    coeff = 1
                    for k in range(self.leads):
                        if k != j:
                            coeff = coeff * (1-preds[i][k][1])

                    #detach coefficient from computation graph
                    coeff=coeff.detach()

                    #calculate loss, add
                    loss1 += (-1.0)*torch.pow( coeff, self.beta/(self.leads-1) )*torch.log(preds[i][j][1])
                    
                    #loss1 += (-1.0)*torch.log(preds[i][j][1])
                    
                #calculate class loss 2
                for j in range(self.leads):
                    
                    #if preds[i][j][1] != 1 and preds[i][j][1] != 0:
                    #compute weighting coefficient
                    coeff = 1
                    for k in range(self.leads):
                        if k != j:
                            coeff = coeff * (1-preds[i][k][2])

                    #detach coefficient from computation graph
                    coeff=coeff.detach()

                    #calculate loss, add
                    loss2 += (-1.0)*torch.pow( coeff, self.beta/(self.leads-1) )*torch.log(preds[i][j][2])
                    
                    #loss1 += (-1.0)*torch.log(preds[i][j][1])
                
                
                
                
                
                
                
                #combine losses into 1x3 tensor
                tmp = torch.stack((loss0,loss1,loss2), 0)
                #print(tmp)
                
                pred_tmp.append(tmp)
            
            #stack all losses together as tensor
            preds = torch.stack(pred_tmp)
            
            return preds


    
################################### WAVEFUSION WLCNN CONTAINER OBJECT ###################################
        
class AttrProxy(object):
    """indexes Wave_Lead_Conv models as Wave_Lead_Conv0, Wave_Lead_Conv1,...
    Wave_Lead_Conv63  in the Wave_Fusion_Model."""
    def __init__(self, module, prefix):
        """
        args:
            module: the Wave_Lead_Conv component to be named
            prefix: int
        """
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        """retrieves the ith Wave_Lead_Conv from Wave_Fusion_Model."""
        return getattr(self.module, self.prefix + str(i))
    

    
    
################################### WLCNN MODEL ###################################    
    
class Wave_Lead_Conv(nn.Module):
    """
    Wave_Lead_Conv is a convolution model designed to convolve over a single scaleogram,
    of shape (1,freq,time) generated by one EEG lead. this model is the portion of 
    onv2d_by_Leads model without the last linear layer that combines the output for each
    lead.
    """
    def __init__(self):
        super(Wave_Lead_Conv,self).__init__()
            
        self.conv1 = nn.Conv2d(1,8, kernel_size=(3,4), stride=(1,2), padding=(1,2)) 
        self.maxPool1 = nn.MaxPool2d((2,2))
      
        self.conv2 = nn.Conv2d(8,16, kernel_size=(4,3), stride=(2,2), padding=1)
        self.maxPool2 = nn.MaxPool2d((2,2))
        
        
        self.conv3 = nn.Conv2d(16,32, kernel_size=(3,4), stride=(1,2), padding=1)
        self.maxPool3 = nn.MaxPool2d((2,2))
        #self.conv3_bn = nn.BatchNorm2d(32)
        
        self.conv4 = nn.Conv2d(32,64, kernel_size=(2,4), stride=(1,1))
        #self.conv4_bn = nn.BatchNorm2d(64)
        
        #self.dropout = nn.Dropout(p=0.50)
        self.fc1 = nn.Linear(64,3) 
   
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self,x):
        #import pdb; pdb.set_trace()
        #convolve over channels only
        x = self.conv1(x)
        x = self.maxPool1(x)
        #x = self.conv1_bn(x)
        x = torch.relu(x)
        
        x = self.conv2(x)
        x = self.maxPool2(x)
        #x = self.conv2_bn(x)
        x = torch.relu(x)
        
        x = self.conv3(x)
        x = self.maxPool3(x)
        #x = self.conv3_bn(x)
        x = torch.relu(x)
        
        x = self.conv4(x)
        #x = self.conv4_bn(x)
        
        #x = self.dropout(x)
        x = x.view(64)
        x = self.fc1(x) 
        x = self.softmax(x)
        #print(x)
        return x
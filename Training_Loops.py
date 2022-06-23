
import copy
import time
import torch
from tqdm.notebook import trange, tqdm

################################### WaveFusion Training Loop ###################################
#set to train w/ GPU if available else cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_WaveFusion_model(model, dataloaders, optimizer, wts_path, beta, delta, epochs=25, load_wts = False, save_as = None, save = False):
    """ train a model with given params
    Args:
        model: model, extends torch.nn
        dataloaders: dataloader dictionary of the form {"train": dataloader_train_data
                                                        "val": dataloader_val_data
                                                        }
        optimizer: optimization func.
        wts_path: path to torch.nn.Module.load_state_dict for "model"
        epochs: number of epochs to train model
        load_wts: bool true if loading a state dict, false otherwhise
        beta: Regularization exponent for Multiplicative Fusion
        delta: delta for boosted fusion loss.
        
    Return:
        Tuple: model with trained weights and validation training statistics(epoch loss, accuracy)
    """
    val_history = []
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_optim = copy.deepcopy(optimizer.state_dict())
    #load moadel weigthts
    if load_wts == True:
        checkpoint = torch.load(wts_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_acc = checkpoint['acc']
        print("weights loaded from" + path_wts)
        print(best_acc)
    #f = open(save_as[:-2]+"txt", "w")

    #train model
    print("num training points  : {}".format( len(dataloaders["train"].dataset)))
    print("num validation points: {}".format( len(dataloaders["val"].dataset)))
    for epoch in range(epochs):
        #import pdb; pdb.set_trace()
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        #f.write('epoch {}/{}'.format(epoch, num_epochs - 1) + "\n")
        print('-' * 10)
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            running_loss = 0.0
            running_corrects = 0 

            for batch in tqdm(dataloaders[phase],desc='batch', leave = False,position=0):
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                optimizer.zero_grad()
                        
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    model.train() #ensure model is in train model

                    # Get model outputs and calculate loss
                    if phase == 'train':
                        #calculate loss
                        preds = model(inputs)
                        loss = boosted_fusion_loss(preds, labels, beta = beta, delta=delta)
                        
                        #get predictions
                        model.eval()
                        preds = model(x = inputs)
                        _, preds = torch.min(preds, 1)
                        
                    else:
                        #calculate loss
                        preds = model(inputs)
                        loss = boosted_fusion_loss(preds, labels, beta = beta, delta=delta)

                        #get predictions
                        model.eval()
                        preds = model(x = inputs)
                        _, preds = torch.min(preds, 1)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        #with autograd.detect_anomaly():
                        if loss != 0:
                            loss.backward()
                            #model_grad_nan(model)
                            optimizer.step()
                    #print(preds)
                    #print(labels)
                    
                    #running statistics
                    if loss != 0:
                        running_loss += loss.item()
                    running_corrects += torch.sum(preds == labels.data)

            time_elapsed = time.time() - since  
            epoch_loss = running_loss 
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                
            #print and log stats
            print('{} epoch_loss:{:.10f} acc:{:.4f} time:{:.4f}'.format(phase, epoch_loss, epoch_acc, time_elapsed))
            #f.write('{} epoch_loss:{:.10f} acc:{:.4f} time:{:.4f}'.format(phase, epoch_loss, epoch_acc, time_elapsed) + "\n")
                     
            #track validation loss and acc
            if phase == 'val':
                val_history.append((epoch_loss,epoch_acc))
                
                if epoch_acc > best_acc :
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_optim = copy.deepcopy(optimizer.state_dict())
                    best_acc = epoch_acc
                
        if epoch % 1 == 0 and epoch != 0 or epoch == epochs-1:
            torch.save({
            'epoch': epoch,
            'acc' : best_acc,
            'model_state_dict': copy.deepcopy(best_model_wts),
            'optimizer_state_dict': copy.deepcopy(best_optim),
            }, save_as+"_ep={}.tar".format(epoch))  

    return model, val_history


################################### CNN Training Loop ###################################

def train_model(model, dataloaders, lossfun, optimizer, wts_path, save_as = None, save = False, epochs=25, load_wts = False):
    """ train a model with given params
    Args:
        model: model, extends torch.nn
        dataloaders: dataloader dictionary of the form {"train": dataloader_train_data
                                                        "val": dataloader_val_data
                                                        }
        lossfun: Loss function
        optimizer: optimization func.
        wts_path: path to torch.nn.Module.load_state_dict for "model"
        epochs: number of epochs to train model
        load_wts: bool true if loading a state dict, false otherwhise
        epochs: number of epochs to train model
        load_wts: bool true if loading a state dict, false otherwhise
        
    
    Return:
        Tuple: model with trained weights and validation training statistics(epoch loss, accuracy)
    """
    
    #isntantiate validation history, base model waits and loss
    val_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    #load moadel weigthts
    if load_wts == True:
        print("loading from: "+path_wts)
        checkpoint = torch.load(path_wts)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    #train model
    print("num training points  : {}".format( len(dataloaders["train"].dataset)))
    print("num validation points: {}".format( len(dataloaders["val"].dataset)))
    
    for epoch in tqdm(range(epochs),desc='epoch', leave = False):
        #import pdb; pdb.set_trace()
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode 

            running_loss = 0.0
            running_corrects = 0  
            
            for batch in tqdm(dataloaders[phase],desc='batch', leave = False):
                #send inputs and labels to device
                inputs = batch[0].to(device)
                labels = batch[1].to(device)

                #clear gradients rom previous batch
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    # Get model outputs and calculate loss for train
                    if phase == 'train':
                        preds = model(inputs)
                        #print(preds) 
                        #print(labels)
                        loss = lossfun(preds, labels)
                        
                        
                    # Get model outputs and calculate loss for val
                    else:
                        preds = model(inputs)
                        loss = lossfun(preds, labels) 

                    #get predictions
                    _, preds = torch.max(preds, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        #back propagate loss
                        loss.backward()
                        #update weights
                        optimizer.step()

                    #running statistics       
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            time_elapsed = time.time() - since

            #update epoch loss and acc
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                  
            #track validation loss and acc
            tqdm.write('{}: {} epoch_loss: {:.10f} epoch_acc: {:.4f} time: {:.4f}'.format(epoch,phase, epoch_loss, epoch_acc,time_elapsed))
            
            #update training history
            if phase == 'val':
                val_history.append(epoch_loss)
            #update best weights    
            if phase == 'val' and best_acc < epoch_acc:
                print("best model updated")
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_optim = copy.deepcopy(optimizer.state_dict())
            
        #save model
        if epoch ==epochs-1 and save:
            torch.save({
            'best_acc': best_acc,
            'model_state_dict': best_model_wts,
            'optimizer_state_dict': best_optim,
            'best_acc': best_acc,
            }, save_as+"_ep={}.tar".format(epoch))  

    #if running in jupyter notebook may want to print in console
    #or refresh cell with each print. Printing too many lines in cell will
    #corrupt the notebook
    model.load_state_dict(best_model_wts)
    return model, val_history




def boosted_fusion_loss(preds, labels, beta, delta):
    """impements boosted loss by Implements multiplicative fusion loss by Liu
    et al https://arxiv.org/pdf/1805.11730.pdf
    args:
        preds: predictions returned by wavelet_fusion model: a list of size ||batchsize||
        labels: labels for datapoint list of size ||batchsize||
        b: scaling factor 0<= b 0<=1
        delta: difference factor 0 <= delta
    Return:
        1-d tensor that can be back propagated.
    """
    channels = 61
    loss = 0
    
    for i in range(len(labels)):
        tmp_loss0 = torch.FloatTensor([0]).to(device)
        tmp_loss1 = torch.FloatTensor([0]).to(device)
        tmp_loss2 = torch.FloatTensor([0]).to(device)
        
        if labels[i] == 0: #class loss for t=0
            
            #compute loss for p^0
            for j in range(channels):
                #print(preds[i][j][0])
                
                if preds[i][j][0] != 1 and preds[i][j][0] != 0: 
                    coeff = 1

                    for k in range(channels):
                        #compute weighting coefficient
                        if k != j:
                            coeff = coeff * (1-preds[i][k][0])

                    #detach coefficient from computation graph
                    coeff=coeff.detach()

                    #calculate loss, add
                    tmp_loss0 += (-1.0)*torch.pow( coeff, beta/(channels-1) )*torch.log(preds[i][j][0])
            
            
            
            #compute loss for p^1
            for j in range(channels):
                #print(preds[i][j][1])
                
                if preds[i][j][1] != 1 and preds[i][j][1] != 0:
                    coeff = 1

                    #compute weighting coefficent
                    for k in range(channels):
                        if k != j:
                            coeff = coeff * (1-preds[i][k][1])

                    #detach coefficient from computation graph
                    coeff=coeff.detach()

                    #calculate loss, add
                    tmp_loss1 += (-1.0)*torch.pow( coeff, beta/(channels-1) )*torch.log(preds[i][j][1])
            

            #compute loss for p^2
            for j in range(channels):
                #print(preds[i][j][1])
                
                if preds[i][j][2] != 1 and preds[i][j][2] != 0:
                    coeff = 1

                    #compute weighting coefficent
                    for k in range(channels):
                        if k != j:
                            coeff = coeff * (1-preds[i][k][2])

                    #detach coefficient from computation graph
                    coeff=coeff.detach()

                    #calculate loss, add
                    tmp_loss2 += (-1.0)*torch.pow( coeff, beta/(channels-1) )*torch.log(preds[i][j][2])
            
            
            
            #update loss
            if tmp_loss0 + delta >= tmp_loss1 or tmp_loss0 + delta >= tmp_loss2:
                #compute loss for p^0
                loss += tmp_loss0
        
        
        
        
        elif labels[i] == 1: #class loss for t=1

            #compute loss for p^0
            for j in range(channels):
                #print(preds[i][j][0])
                
                if preds[i][j][0] != 1 and preds[i][j][0] != 0: 
                    coeff = 1

                    for k in range(channels):
                        #compute weighting coefficient
                        if k != j:
                            coeff = coeff * (1-preds[i][k][0])

                    #detach coefficient from computation graph
                    coeff=coeff.detach()

                    #calculate loss, add
                    tmp_loss0 += (-1.0)*torch.pow( coeff, beta/(channels-1) )*torch.log(preds[i][j][0])
            
            
            
            #compute loss for p^1
            for j in range(channels):
                #print(preds[i][j][1])
                
                if preds[i][j][1] != 1 and preds[i][j][1] != 0:
                    coeff = 1

                    #compute weighting coefficent
                    for k in range(channels):
                        if k != j:
                            coeff = coeff * (1-preds[i][k][1])

                    #detach coefficient from computation graph
                    coeff=coeff.detach()

                    #calculate loss, add
                    tmp_loss1 += (-1.0)*torch.pow( coeff, beta/(channels-1) )*torch.log(preds[i][j][1])
                    
            
            #compute loss for p^2
            for j in range(channels):
                #print(preds[i][j][1])
                
                if preds[i][j][2] != 1 and preds[i][j][2] != 0:
                    coeff = 1

                    #compute weighting coefficent
                    for k in range(channels):
                        if k != j:
                            coeff = coeff * (1-preds[i][k][2])

                    #detach coefficient from computation graph
                    coeff=coeff.detach()

                    #calculate loss, add
                    tmp_loss2 += (-1.0)*torch.pow( coeff, beta/(channels-1) )*torch.log(preds[i][j][2])            


            if tmp_loss1 + delta >= tmp_loss0 or tmp_loss1 + delta >= tmp_loss2:
                loss += tmp_loss1
                
        elif labels[i] == 2:
            

            #compute loss for p^0
            for j in range(channels):
                #print(preds[i][j][0])
                
                if preds[i][j][0] != 1 and preds[i][j][0] != 0: 
                    coeff = 1

                    for k in range(channels):
                        #compute weighting coefficient
                        if k != j:
                            coeff = coeff * (1-preds[i][k][0])

                    #detach coefficient from computation graph
                    coeff=coeff.detach()

                    #calculate loss, add
                    tmp_loss0 += (-1.0)*torch.pow( coeff, beta/(channels-1) )*torch.log(preds[i][j][0])
            
            
            
            #compute loss for p^1
            for j in range(channels):
                #print(preds[i][j][1])
                
                if preds[i][j][1] != 1 and preds[i][j][1] != 0:
                    coeff = 1

                    #compute weighting coefficent
                    for k in range(channels):
                        if k != j:
                            coeff = coeff * (1-preds[i][k][1])

                    #detach coefficient from computation graph
                    coeff=coeff.detach()

                    #calculate loss, add
                    tmp_loss1 += (-1.0)*torch.pow( coeff, beta/(channels-1) )*torch.log(preds[i][j][1])
                    
            
            #compute loss for p^2
            for j in range(channels):
                #print(preds[i][j][1])
                
                if preds[i][j][2] != 1 and preds[i][j][2] != 0:
                    coeff = 1

                    #compute weighting coefficent
                    for k in range(channels):
                        if k != j:
                            coeff = coeff * (1-preds[i][k][2])
                    coeff=coeff.detach()

                    #calculate loss, add
                    tmp_loss2 += (-1.0)*torch.pow( coeff, beta/(channels-1) )*torch.log(preds[i][j][2])            


            if tmp_loss2 + delta >= tmp_loss0 or tmp_loss1 + delta >= tmp_loss1:
                loss += tmp_loss2            
           
    return loss
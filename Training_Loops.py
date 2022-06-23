
import copy
import time
import torch
from tqdm.notebook import trange, tqdm

################################### WaveFusion Training Loop ###################################
#set to train w/ GPU if available else cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, dataloaders, loss, optimizer, save_as = None,save = False, epochs=None, load_wts = False, wts = None):
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
        
    Return:
        Tuple: model with trained weights and validation training statistics(epoch loss, accuracy)
    """
    
    #isntantiate validation history, base model waits and loss
    val_loss_history = []
    train_loss_history = []

    val_acc_history = []
    train_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_optim = None
    #load moadel weigthts
    if load_wts == True:
        print("loading from: "+path_wts)
        checkpoint = torch.load(path_wts)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("acc from prev:{:.4f}".format(checkpoint['best_acc']))
    
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
            
            for batch in tqdm(dataloaders[phase],desc='batches', leave = False):
                #send inputs and labels to device
                inputs = batch[0].to(device)
                labels = batch[1].to(device)

                #clear gradients rom previous batch
                optim.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    # Get model outputs and calculate loss for train
                    if phase == 'train':
                        preds = model(inputs)
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
                        optim.step()

                    #running statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            time_elapsed = time.time() - since

            #update epoch loss and acc
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                  
            #track validation loss and acc
            print('{}: {} epoch_loss: {:.10f} epoch_acc: {:.4f} time: {:.4f}'.format(epoch,phase, epoch_loss, epoch_acc,time_elapsed))
            
            #update training history
            if phase == 'val':
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
            #update best weights
            if phase == 'val' and best_acc < epoch_acc:
                print("best model updated")
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_optim = copy.deepcopy(optimizer.state_dict())
            
        #save model
        if (epoch ==epochs-1 and save) or (epoch % 4 == 0 and epoch != 0 and save == True):
            torch.save({
            'best_acc': best_acc,
            'model_state_dict': best_model_wts,
            'optimizer_state_dict': best_optim,
            'best_acc': best_acc,
            }, save_as+"_ep={}.tar".format(epoch))

    model.load_state_dict(best_model_wts)
    print(best_acc)
    history = (val_loss_history, val_acc_history, train_loss_history, train_acc_history)
    return model, history

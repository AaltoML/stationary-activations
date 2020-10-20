import pandas as pd
import platform
if platform.system() == "Darwin":
    print("MacOSX detected, switching matplotlib backend.")
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from tqdm import tqdm
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Set random seeds for reproducibility
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def sigma(x, device, nu_ind=1, ell=0.5):
    """Implements the Matern activation function denoted as sigma(x) in Equation 9.
    sigma(x) corresponds to a Matern kernel, with specified smoothness
    parameter nu and length-scale parameter ell.
    
    Args:
      x: Input to the activation function
      device: A torch.device object
      nu_ind: Index for choosing Matern smoothness (look at nu_list below)
      ell: Matern length-scale, only 0.5 and 1 available with precalculated scaling coefficients
    """
    nu_list = [1/2, 3/2, 5/2, 7/2, 9/2] #list of available smoothness parameters
    nu = torch.tensor(nu_list[nu_ind]).to(device) #smoothness parameter
    lamb =  torch.sqrt(2*nu)/ell  #lambda parameter
    v = nu+1/2
    # Precalculated scaling coefficients for two different lengthscales (q divided by Gammafunction at nu + 0.5)
    ell05A = [4.0, 19.595917942265423, 65.31972647421809, 176.69358285524189, 413.0710073859664]
    ell1A = [2.0, 4.898979485566356, 8.16496580927726, 11.043348928452618, 12.90846898081145]
    if ell == 0.5:
        A = torch.tensor(ell05A[nu_ind]).to(device)
    if ell == 1:
        A = torch.tensor(ell1A[nu_ind]).to(device)
    y = A*torch.sign(x)*torch.abs(x)**(v-1)*torch.exp(-lamb*torch.abs(x))
    y[x<0] = 0
    return y

################### Network architecture #########################################
class MLP(nn.Module):
    def __init__(self, num_features, num_classes=2, dropout=0.2, device='cpu'):
        super(MLP, self).__init__()

        #FC layers
        self.fc1 = nn.Linear(num_features, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 500)
        self.fc4 = nn.Linear(500, 50)
        self.fc5 = nn.Linear(50, num_classes)
        self.drop_layer = nn.Dropout(p=dropout)
        self.device1 = device

    def forward(self, x):

        #FC branch
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x3 = F.relu(self.fc3(x2))
        x4 = sigma(self.fc4(x3), self.device1)
        x5 = self.drop_layer(x4)
        y = self.fc5(x5)
        return y

#Function for applying dropout in model evaluation mode
def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()

###########################################################

# Dataset loading function
def load_dataset(full_path): 
    dataframe = pd.read_csv(full_path, header=None, na_values='?') # load the dataset as a numpy array
    dataframe = dataframe.dropna() # drop rows with missing values
    last_ix = len(dataframe.columns) - 1 
    X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix] # split into inputs and outputs
    # select categorical and numerical features
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    num_ix = X.select_dtypes(include=['int64', 'float64']).columns
    # label encode the target variable to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)
    return X.values, y, cat_ix, num_ix

#Specify dataset specific parameters
datasets = ['diabetes', 'adult']
LRlist = [0.0001, 0.00005] #individual learning rates for datasets

fig, ax = plt.subplots(nrows=2, ncols=len(datasets)) #Figure and axes for plotting mid training results

for dataset_ind in range(len(datasets)):  #Loop through datasets

    data_name = datasets[dataset_ind]
    num_classes = 2
    method_name = 'DNN-matern32'

    n_splits=10 #number of splits in K-fold cross-validation

    LR = LRlist[dataset_ind]
    n_epochs = 20
    milestones=[0.5 * n_epochs, 0.75 * n_epochs] #milestones for decaying learning rate
    batch_size = 500

    MC = 100
    dropout = 0.2
     
    # define the location of the dataset
    full_path = 'data/{}.csv'.format(data_name)

    # load the dataset
    X, y, cat_ix, num_ix = load_dataset(full_path)

    # Get categories for one-hot encoding for consistency accross folds
    enc = OneHotEncoder(handle_unknown='ignore')
    onehotenc = enc.fit(X)
    categories = [enc.categories_[i] for i in cat_ix.values]

    # define preprocessing steps (one-hot encoding + standard scaling)
    steps = [('c',OneHotEncoder(handle_unknown='ignore', categories = categories),cat_ix), ('n',StandardScaler(),num_ix)]

    # Generate transformer for one-hot encoding categorical and normalizing numerical features
    ct = ColumnTransformer(steps, sparse_threshold=0)
    N = y.shape[0] #Number of datapoints
    D = X.shape[1] #Number of features

    # Train on cuda if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #Initialize arrays for saving results
    accuracies = np.zeros((n_splits, n_epochs))
    losses = np.zeros((n_splits, n_epochs))
    NLLLosses = np.zeros((n_splits))

    # Shuffle data once before splitting
    shuffle_ind = np.arange(0,N)
    random.shuffle(shuffle_ind)

    for split in tqdm(range(n_splits)): #Loop over the folds
        #Get train/test indexes for the current split
        cut_start = split*int(np.floor(N/n_splits))
        cut_end = (split+1)*int(np.floor(N/n_splits))
        test_ind = shuffle_ind[cut_start:cut_end]
        train_ind = np.hstack((shuffle_ind[0:cut_start], shuffle_ind[cut_end:]))

        X_test = X[test_ind, :]
        y_test = y[test_ind]
        X_train = X[train_ind, :]
        y_train = y[train_ind]

        norm = ct.fit(X_train) #Fit the preprocessing transform based on training data
        # Preprocess both train and test data with the fitted transform
        X_train_norm = norm.transform(X_train)
        X_test_norm = norm.transform(X_test)
        num_features = X_train_norm.shape[1] #Number of features after preprocessing
        N_train_samples = X_train_norm.shape[0]
        N_test_samples = X_test_norm.shape[0]

        # Transform to torch
        X_train_t = torch.from_numpy(X_train_norm).float()
        X_test_t = torch.from_numpy(X_test_norm).float()

        y_train_t = torch.from_numpy(y_train).long()
        y_test_t = torch.from_numpy(y_test).long()

        net = MLP(num_features, num_classes = num_classes, dropout = dropout, device = device)

        #If GPU resources are available, switch to cuda
        if torch.cuda.is_available():
            net = net.cuda()
            X_train_t = X_train_t.cuda()
            y_train_t = y_train_t.cuda()
            X_test_t = X_test_t.cuda()
            y_test_t = y_test_t.cuda()

        criterion = nn.CrossEntropyLoss() #Training loss
        mc_crit = nn.NLLLoss() #Evaluation loss for NLPD calculations
        optimizer = optim.Adam(net.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

        
        # Train network
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            index = np.arange(0,N_train_samples)
            random.shuffle(index)
            #Shuffle training data accross epochs
            curr_input_set = X_train_t[index,:]
            curr_target_set = y_train_t[index]
            curr_ind = 0
            while curr_ind < N_train_samples: # Loop over minibatches
                new_ind = min(N_train_samples, curr_ind + batch_size)
                outputs = net(curr_input_set[curr_ind:new_ind,:])
                outputs = torch.squeeze(outputs)
                loss = criterion(outputs, curr_target_set[curr_ind:new_ind])
                curr_ind = new_ind
                # zero the parameter gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step() #Update learning rate

            # validation of the epoch
            curr_ind = 0
            hits = 0
            net.eval()
            loss = 0
            counter = 0
            while curr_ind < N_test_samples: #Loop over minibatches
                new_ind = min(N_test_samples, curr_ind + batch_size)
                outputs = net(X_test_t[curr_ind:new_ind,:])
                outputs = torch.squeeze(outputs)
                y_test_curr = y_test[curr_ind:new_ind]
                loss += criterion(outputs, y_test_t[curr_ind:new_ind]).cpu().detach().numpy()
                curr_ind = new_ind
                outputs = torch.squeeze(outputs).cpu().detach().numpy()
                preds = np.argmax(outputs, axis=-1)
                hits += np.sum(y_test_curr.astype(int) == preds)
                counter += 1
            losses[split, epoch] = loss/counter #save loss of the epoch
            accuracies[split, epoch] = hits/N_test_samples #save classification accuracy of the epoch
            net.train()
            

        # Evaluate network
        net.eval()

        net.apply(apply_dropout) #activate dropout for MC dropout sampling

        #MC dropout samples
        MC_outputs = torch.zeros(N_test_samples, num_classes, MC)
        for i in range(MC): #Loop for the number of MC dropout samples
            curr_ind = 0
            while curr_ind < N_test_samples: #Loop over minibatches
                new_ind = min(N_test_samples, curr_ind + batch_size)
                outputs = net(X_test_t[curr_ind:new_ind,:]).cpu().detach()
                MC_outputs[curr_ind:new_ind,:,i] = outputs
                curr_ind = new_ind
        
        MC_logprobs = F.log_softmax(MC_outputs, dim = -2) #calculate class probabilities using softmax function
        mean_MC_logprobs = torch.mean(MC_logprobs, -1) #mean accross MC dropout samples
        nllloss = mc_crit(mean_MC_logprobs, y_test_t.cpu()).numpy() #calculate NLPD (same as NLL-loss)
        NLLLosses[split] = nllloss #Save NLPD value of the split

    #Plot mid-training losses and accuracies for the current dataset
    ax[0,dataset_ind].plot(losses.transpose())
    ax[1,dataset_ind].plot(accuracies.transpose())
    ax[0,dataset_ind].set_title(('losses: '+ datasets[dataset_ind]))
    ax[1,dataset_ind].set_title(('accuracies: '+ datasets[dataset_ind]))
    ax[0,dataset_ind].set_xlabel('epoch number')
    ax[1,dataset_ind].set_xlabel('epoch number')
    ax[0,dataset_ind].set_ylabel('NLPD')
    ax[1,dataset_ind].set_ylabel('accuracy')

    #Calculate mean and std of NLPD and accuracy accross folds, round for printing
    mean_nlpd = np.around(np.mean(NLLLosses),3)
    std_nlpd = np.around(np.std(NLLLosses),3)
    mean_acc = np.around(np.mean(accuracies[:,-1]),3)
    std_acc = np.around(np.std(accuracies[:,-1]),3)
    print('Dataset name: ',datasets[dataset_ind])
    print('method name: ',method_name)
    print('number of samples: ',N)
    print('number of features: ',D)
    print('mean NLPD: ',mean_nlpd)
    print('std of NLPD: ',std_nlpd)
    print('mean ACC: ',mean_acc)
    print('std of ACC: ',std_acc)

fig.tight_layout()
plt.savefig("UCI_benchmark_results.png")
plt.show()

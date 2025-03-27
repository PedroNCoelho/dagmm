# %% [markdown]
# # Deep Autoencoding Gaussian Mixture Model for Unsupervised Anomaly Detection - Adversarial Examples

# %%
import numpy as np
from data_loader import *
from main import *
import time
from IPython.utils import io
import torch

# %%
# ## ========= MAIN EXECUTION VARIABLES ==========

data_path = '../data'
num_examples_per_attack = 1000

### At this threshold the FPR of the model we trained is 0.1
real_thr = 3.3254204
thr = real_thr*0.8

# label_names = ['FTP-Patator','SSH-Patator','DoS slowloris','DoS Slowhttptest','DoS Hulk','DoS GoldenEye',
#           'Heartbleed','Web Attack','Infiltration', 'Bot', 'PortScan', 'DDoS']
label_names = ['FTP-Patator']

# %% [markdown]
# ## Treinamento

# %%
class hyperparams():
    def __init__(self, config):
        self.__dict__.update(**config)
defaults = {
    'lr' : 1e-4,
    'num_epochs' : 200,
    'batch_size' : 1024,
    'gmm_k' : 4,
    'lambda_energy' : 0.1,
    'lambda_cov_diag' : 0.005,
    'pretrained_model' : None,
    'mode' : 'train_only',
    'use_tensorboard' : False,
    'data_path' : data_path,
    'ds': 'cicids2017',

    'log_path' : './dagmm/logs',
    'model_save_path' : './dagmm/models',
    'sample_path' : './dagmm/samples',
    'test_sample_path' : './dagmm/test_samples',
    'result_path' : './dagmm/results',

    'log_step' : None,
    'sample_step' : 194,
    'model_save_step' : 194,
}

# %%
solver = main(hyperparams(defaults))

# %% [markdown]
# ## Inferência

# %% [markdown]
# ### Constantes do teste

# %%
test_files = [data_path+'/flow_based/Tuesday-WH-generate-labeled.csv',
            data_path+'/flow_based/Wednesday-WH-generate-labeled.csv',
            data_path+'/flow_based/Thursday-WH-generate-labeled.csv',
            data_path+'/flow_based/Friday-WH-generate-labeled.csv']

# test_files = [data_path+'/flow_based/Thursday-WH-generate-labeled.csv',
#                 data_path+'/flow_based/Tuesday-WH-generate-labeled.csv']

train_min = np.load(data_path+'/flow_based/x_train_meta/train_min.npy')
train_max = np.load(data_path+'/flow_based/x_train_meta/train_max.npy')

x_test_all = []
y_test_all = []
all_label_set = []
for i in range(len(test_files)):
    print (i,test_files[i])
    url_data = test_files[i]
    df = pd.read_csv(url_data)

    feats = df.iloc[:,8:]
    ds_port = df.iloc[:,5]
    df = pd.concat([ds_port,feats],axis=1)

    labels = df.iloc[:,-1].values
    label_set = set(labels)
    all_label_set.append(label_set)

    all_feats = df.iloc[:,:-1].astype(np.float64).values
    known_data_IDs =(np.any(np.isinf(all_feats),axis=1) + np.any(np.isnan(all_feats),axis=1))==False
    x_test = all_feats[known_data_IDs]
    y_test = df.iloc[:,-1].values
    y_test = y_test[known_data_IDs]
#         x_test = (x_test - train_min)/(train_max - train_min+1e-6)
    x_test_all.append(x_test)
    y_test_all.append(y_test)
x_test = np.concatenate(x_test_all,axis=0).astype(np.float32)
y_test = np.concatenate(y_test_all,axis=0)

#### list of features which are decimal:
decimal_features = []
for i in range(x_test.shape[1]):
    a1 = x_test[:,i]
    a2 = np.round(a1)
    temp = np.sum(np.abs(a1-a2))
    if temp==0:
#             print (i,df.columns[i])
        decimal_features.append(i)

num_input = x_test.shape[1]

# %%
train_min_t = torch.tensor(train_min, dtype=torch.float32)
train_max_t = torch.tensor(train_max, dtype=torch.float32)

# %%
alpha = torch.zeros((1, x_test.shape[1]), requires_grad=True, dtype=torch.float32)

# %%
fixed_mask = np.ones(77,dtype=np.float32)

######### Group 1 features (based on the categorization in  "Towards Evaluation of NIDSs in Adversarial Setting") #########
fixed_mask[0] = 0 #dst_port
fixed_mask[3] = 0 #bwd
fixed_mask[4] = 0 #total len of fwd pkts (sum of all payloads in fwd direction)
fixed_mask[5] = 0 #total len of bwd pkts (sum of all payloads in bwd direction)
fixed_mask[10] = 0 #bwd
fixed_mask[11] = 0 #bwd
fixed_mask[12] = 0 #bwd
fixed_mask[13] = 0 #bwd
fixed_mask[25] = 0 #bwd
fixed_mask[26] = 0 #bwd
fixed_mask[27] = 0 #bwd
fixed_mask[28] = 0 #bwd
fixed_mask[29] = 0 #bwd
fixed_mask[31] = 0 #bwd
fixed_mask[33] = 0 #bwd
fixed_mask[35] = 0 #bwd
fixed_mask[37] = 0 #bwd
fixed_mask[53] = 0
fixed_mask[54] = 0 #bwd
for i in range(58,65):
    fixed_mask[i]=0
fixed_mask[66] = 0 #bwd



######### Group 4 features (based on the categorization in  "Towards Evaluation of NIDSs in Adversarial Setting") #########
fixed_mask[9] = 0 #std (fwd-payload)
fixed_mask[17] = 0 #std
fixed_mask[22] = 0 #std
fixed_mask[41] = 0 #std
fixed_mask[42] = 0 #var
fixed_mask[55] = 0 #bulk
fixed_mask[56] = 0 #bulk
fixed_mask[57] = 0 #bulk
for i in range(68,77):
    fixed_mask[i]=0



################## Group 3 (Dependent) features (based on the categorization in  "Towards Evaluation of NIDSs in Adversarial Setting") #########
depended_features = {8,14,15,16,20,21,34,36,38,39,40,51,52,67}
for i in depended_features:
    fixed_mask[i] = 0


################## Other features are Group 2 (Independent) features


mask_l1 = np.copy(fixed_mask)
mask_l2 = np.copy(fixed_mask)
####unmask stds
mask_l2[9]=1
mask_l2[17]=1
mask_l2[22]=1
mask_l2[41]=1
mask_l2[42]=1
mask_l3 = np.copy(mask_l2)
mask_l3[55] = 1 #bulk
mask_l3[56] = 1 #bulk
mask_l3[57] = 1 #bulk
for i in range(68,77):
    mask_l3[i]=1


#### The duplicated features:
dup_ht={61:2,
        63:3,
        62:4,
        64:5}
mins = [7,19,24,38]
maxs = [6,18,23,39]
flags = [30,32,43,44,45,46,47,48,49,50]
aggr_features=[9,17,22,41,42,55,56,57,68,69,70,71,72,73,74,75,76]

optimizer_001 = torch.optim.Adam([alpha], lr=0.001)
optimizer_01 = torch.optim.Adam([alpha], lr=0.01)
optimizer_1 = torch.optim.Adam([alpha], lr=0.1)
optimizer_10 = torch.optim.Adam([alpha], lr=1.0)
all_optimizers = [optimizer_001,optimizer_01,optimizer_1,optimizer_10]

# %% [markdown]
# ### Funções auxiliares

# %%
def normalize(x):
    x_normalized =(x - train_min_t)/(train_max_t - train_min_t+1e-6)
    return x_normalized

# %%
def get_score(x):
    with io.capture_output() as captured:
        score = solver.test([normalize(x)], energy_only=True)[0]
    return score

# %%
def get_adv(x,adv_mask):
    x_normalized = (x - train_min)/(train_max - train_min+1e-6)
    alpha_masked = alpha*torch.tensor(adv_mask, dtype=torch.float32)
    adv_ex_normalized = torch.tensor(x_normalized, dtype=torch.float32) + alpha_masked
    adv_ex = adv_ex_normalized*(train_max_t - train_min_t + 1e-6) + (train_min_t) #### unnormalized
    return adv_ex.detach()


def optim(x,adv_mask,optimizer):
    x_normalized = (x - train_min)/(train_max - train_min+1e-6)
    alpha_masked = alpha*torch.tensor(adv_mask, dtype=torch.float32)
    adv_ex_normalized = torch.tensor(x_normalized, dtype=torch.float32) + alpha_masked

    with io.capture_output() as captured:
      score = solver.test([adv_ex_normalized], energy_only=True)[0]

    loss = torch.tensor(score, requires_grad=True, dtype=torch.float32)

    loss.backward()  # Calculate gradients
    optimizer.step()  # Update alpha

# %%
def find_adv(source_index, x_test_mal, mal_counter):
    alpha.data.zero_()
    ##### CHECK TO SEE IF IT NEEDS TO BE ADVERSARIALLY CHANGED #####
    orig_sample = np.copy(x_test_mal[source_index][None])
    adv_ex = get_adv(orig_sample,mask_l1)
    sc = get_score(adv_ex)
    if sc<thr:
        return 'no change'
    mal_counter[0]+=1
    backup_adv = [None]
    def optimize(optimizer,total_iter,mask_v):
        alpha.data.zero_()
        for i in range(total_iter):
            optim(orig_sample,mask_v,optimizer)
            adv_ex = get_adv(orig_sample,mask_v)
            adv_ex_np = adv_ex.numpy()
            for k in dup_ht:
                adv_ex_np[0,k] = adv_ex_np[0,dup_ht[k]]
            adv_ex_np[0,mins] = np.maximum(0,adv_ex_np[0,mins])
            adv_ex_np[0,mins] = np.minimum(orig_sample[0,mins],adv_ex_np[0,mins])
            adv_ex_np[0,maxs] = np.maximum(orig_sample[0,maxs],adv_ex_np[0,maxs])
            adv_ex_np[0,flags] = np.maximum(orig_sample[0,flags],adv_ex_np[0,flags])
            flags_max_changed = np.max(adv_ex_np[0,flags] - orig_sample[0,flags])
            adv_ex_np[0,aggr_features] = np.maximum(0,adv_ex_np[0,aggr_features])
            adv_ex_np[0,2] = np.maximum(orig_sample[0,2]+flags_max_changed,adv_ex_np[0,2])
            adv_ex_np[0,65] = np.maximum(0,adv_ex_np[0,65])

            ##### round the ones that should be rounded ####
            adv_ex_np[0,decimal_features] = np.round(adv_ex_np[0,decimal_features])
            ##################### recalculate dependent features ######################
            adv_ex_np[0,4] = adv_ex_np[0,4] + (adv_ex_np[0,6]!=orig_sample[0,6])*adv_ex_np[0,6] + (adv_ex_np[0,7]!=orig_sample[0,7])*adv_ex_np[0,7]
            adv_ex_np[0,8] = adv_ex_np[0,4]/adv_ex_np[0,2]
            adv_ex_np[0,14]=(adv_ex_np[0,4]+adv_ex_np[0,5])/adv_ex_np[0,1]*1e6
            adv_ex_np[0,15]=(adv_ex_np[0,2]+adv_ex_np[0,3])/adv_ex_np[0,1]*1e6
            adv_ex_np[0,16]=adv_ex_np[0,1]/(adv_ex_np[0,2]+adv_ex_np[0,3]-1)

            adv_ex_np[0,21]=adv_ex_np[0,20]/(adv_ex_np[0,2]-1)
            adv_ex_np[0,34]=adv_ex_np[0,34] + 20*(adv_ex_np[0,2] - orig_sample[0,2])
            adv_ex_np[0,36]=adv_ex_np[0,2]/adv_ex_np[0,1]*1e6

            adv_ex_np[0,38]=np.minimum(adv_ex_np[0,38],adv_ex_np[0,7])
            adv_ex_np[0,39]=np.maximum(adv_ex_np[0,39],adv_ex_np[0,6])

            adv_ex_np[0,40]=(adv_ex_np[0,4]+adv_ex_np[0,5])/(adv_ex_np[0,3]+adv_ex_np[0,2]+1)
            adv_ex_np[0,51]=adv_ex_np[0,3]/adv_ex_np[0,2]
            adv_ex_np[0,52]=(adv_ex_np[0,4]+adv_ex_np[0,5])/(adv_ex_np[0,3]+adv_ex_np[0,2])
            adv_ex_np[0,53]=adv_ex_np[0,8]
            adv_ex_np[0,67]=adv_ex_np[0,67] + (adv_ex_np[0,6]!=orig_sample[0,6])*1 + (adv_ex_np[0,7]!=orig_sample[0,7])*1

            adv_ex_np[np.isinf(adv_ex_np)]=0
            adv_ex_np[np.isnan(adv_ex_np)]=0
            adv_ex_t = torch.tensor(adv_ex_np, dtype=torch.float32)
            sc = get_score(adv_ex_t)
            if sc<thr:
                return adv_ex_np
            if sc<real_thr and isinstance(backup_adv[0],type(None)):
                backup_adv[0] = adv_ex_np

        return None

    for mask in [mask_l1,mask_l2,mask_l3]:
        res = optimize(optimizer_001,30,mask)
        if isinstance(res,type(None)):
            res = optimize(optimizer_01,40,mask)
            if isinstance(res,type(None)):
                res = optimize(optimizer_1,50,mask)
                if isinstance(res,type(None)):
                    res = optimize(optimizer_10,60,mask)
        if isinstance(res,type(None))==False:
            break
    if isinstance(res,type(None)):
        return backup_adv[0]
    return res


# %% [markdown]
# ### Exemplo adversarial por tipo de ataque

# %%
for attack_type in label_names:

    print("#####################" + attack_type + "######################")

    x_test_mal = x_test[y_test==attack_type]
    if(len(x_test_mal) != 0):
        x_test_mal = x_test_mal[:num_examples_per_attack].astype(np.float32)
        x_test_adv = np.zeros_like(x_test_mal)
        mal_counter = [0]
        cons_as_mal = 0
        cons_as_ben = 0
        fooled = 0
        st = time.time()
        print (attack_type, x_test_mal.shape)
        for i in range(len(x_test_mal)):
            res = find_adv(i, x_test_mal, mal_counter)
            if isinstance(res, str) and res =='no change':
                cons_as_ben+=1
                x_test_adv[i] = np.copy(x_test_mal[i].astype(np.float32))
            elif isinstance(res,type(None)):
                cons_as_mal+=1
                x_test_adv[i] = np.copy(x_test_mal[i].astype(np.float32))
            else:
                fooled+=1
                x_test_adv[i] = res

        print ("TPR of the attacker's local copy of the NIDS: {0:0.4f}".format(cons_as_mal/len(x_test_mal)))


        score_np2 = np.zeros(len(x_test_adv))
        begin_time = time.time()
        for i in range(len(x_test_adv)):
            sample = x_test_adv[i][None]
            sample_t = torch.from_numpy(sample)
            score_temp = get_score(sample_t)
            score_np2[i] = score_temp
        mal_scores = score_np2
        print ("TPR of the victim's NIDS: {0:0.4f}".format(np.sum(mal_scores>=thr)/(0. + len(mal_scores))))



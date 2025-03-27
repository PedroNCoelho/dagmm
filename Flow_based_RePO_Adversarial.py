import tensorflow as tf
import numpy as np
import os
import sys
import time
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import Model
import pandas as pd


### At this threshold the FPR of the model we trained is 0.1
real_thr = 0.01653931848704815
### We consider a threshold lower than the real threshold of the model to compensate the impact of group 4 features
### as described in "Towards Evaluation of NIDSs in Adversarial Setting"
thr = real_thr*0.8


def get_test_set():
    test_files = ['../data/flow_based/Tuesday-WH-generate-labeled.csv',
                '../data/flow_based/Wednesday-WH-generate-labeled.csv',
                '../data/flow_based/Thursday-WH-generate-labeled.csv',
                '../data/flow_based/Friday-WH-generate-labeled.csv']

    train_min = np.load('../data/flow_based/x_train_meta/train_min.npy')
    train_max = np.load('../data/flow_based/x_train_meta/train_max.npy')

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
    return x_test, y_test,train_min, train_max,decimal_features



@tf.function
def test_step(x):
    def_mask = tf.random.uniform(shape=[1*100,num_input])
    def_mask = tf.cast((def_mask>0.75),tf.float32)
    x_normalized =(x - train_min)/(train_max - train_min+0.000001)

    partial_x = def_mask*x_normalized
    rec_x = model(partial_x, training=False)
    score = tf.reduce_mean(tf.square(rec_x - x_normalized),axis=1)
    score = tf.reduce_min(tf.reshape(score,[5,20]),axis=-1)
    score = tf.reduce_sum(score)
    return score

label_names = ['DoS Slowhttptest']

for attack_type in label_names:

    print("#####################" + attack_type + "######################")
    model = tf.keras.models.load_model('../models/flw_model/')

    x_test, y_test, train_min, train_max, decimal_features = get_test_set()
    num_input = x_test.shape[1]

    x_test_mal = x_test[y_test==attack_type]
    print (x_test_mal.shape)
    x_test_mal = x_test_mal[:1000].astype(np.float32)
    score_np = np.zeros(len(x_test_mal))
    begin_time = time.time()
    for i in range(len(x_test_mal)):
        sample = x_test_mal[i][None]
        score_temp = test_step(sample)
        score_np[i] = score_temp.numpy()
    print (i,time.time() - begin_time)

    print ("TPR in normal setting for "+attack_type+" is {0:0.4f}".format(np.sum(score_np>=real_thr)/len(score_np)))


    # Crafting Adversarial Examples:

    alpha = tf.Variable(np.zeros((1, x_test.shape[1]), dtype=np.float32),name='modifier')

    @tf.function
    def get_adv(x,adv_mask):
        x_normalized = (x - train_min)/(train_max - train_min+1e-6)
        alpha_masked = alpha*adv_mask
        adv_ex_normalized = x_normalized + alpha_masked
        adv_ex = adv_ex_normalized*(train_max - train_min + 1e-6) + train_min #### unnormalized
        return adv_ex

    @tf.function
    def optim(x,adv_mask,optimizer):
        with tf.GradientTape() as tape:
            x_normalized = (x - train_min)/(train_max - train_min+1e-6)
            alpha_masked = alpha*adv_mask
            adv_ex_normalized = x_normalized + alpha_masked

            def_mask = tf.random.uniform(shape=[1*100,num_input])
            def_mask = tf.cast((def_mask>0.75),tf.float32)
            partial_x = def_mask*adv_ex_normalized
            rec_x = model(partial_x, training=False)

            score = tf.reduce_mean(tf.square(rec_x - adv_ex_normalized),axis=1)
            score = tf.reduce_sum(score)
            loss = score

        gradients = tape.gradient(loss, [alpha])
        optimizer.apply_gradients(zip(gradients, [alpha]))


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


    optimizer_001 = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizer_01 = tf.keras.optimizers.Adam(learning_rate=0.01)
    optimizer_1 = tf.keras.optimizers.Adam(learning_rate=0.1)
    optimizer_10 = tf.keras.optimizers.Adam(learning_rate=1.)
    all_optimizers = [optimizer_001,optimizer_01,optimizer_1,optimizer_10]
    for op in all_optimizers:
        op.apply_gradients(zip([alpha],[alpha]))


    def find_adv(source_index):
        alpha.assign(np.zeros(alpha.shape))
        ##### CHECK TO SEE IF IT NEEDS TO BE ADVERSARIALLY CHANGED #####
        orig_sample = np.copy(x_test_mal[source_index][None])
        adv_ex = get_adv(orig_sample,mask_l1)
        sc = test_step(adv_ex)
        sc = sc.numpy()
        if sc<thr:
            return 'no change'
        mal_counter[0]+=1
        backup_adv = [None]
        def optimize(optimizer,total_iter,mask_v):
            alpha.assign(np.zeros(alpha.shape))
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
                sc = test_step(adv_ex_np)
                sc = sc.numpy()
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



    x_test_mal = x_test[y_test==attack_type]
    x_test_mal = x_test_mal[:1000].astype(np.float32)
    x_test_adv = np.zeros_like(x_test_mal)
    mal_counter = [0]
    cons_as_mal = 0
    cons_as_ben = 0
    fooled = 0
    st = time.time()
    print (attack_type, x_test_mal.shape)
    for i in range(len(x_test_mal)):
        res = find_adv(i)
        if res=='no change':
            cons_as_ben+=1
            x_test_adv[i] = np.copy(x_test_mal[i])
        elif isinstance(res,type(None)):
            cons_as_mal+=1
            x_test_adv[i] = np.copy(x_test_mal[i])
        else:
            fooled+=1
            x_test_adv[i] = res


    print ("TPR of the attacker's local copy of the NIDS: {0:0.4f}".format(cons_as_mal/len(x_test_mal)))


    score_np2 = np.zeros(len(x_test_adv))
    begin_time = time.time()
    for i in range(len(x_test_adv)):
        sample = x_test_adv[i][None]
        score_temp = test_step(sample)
        score_np2[i] = score_temp.numpy()
    mal_scores = score_np2
    print ("TPR of the victim's NIDS: {0:0.4f}".format(np.sum(mal_scores>=thr)/(0. + len(mal_scores))))




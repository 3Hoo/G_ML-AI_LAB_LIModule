import os, sys
import argparse
import numpy as np
import time
import torch
from torch.nn.functional import cross_entropy
import torch.optim as optim
from loss import LogManager, calc_gaussprob, calc_kl_vae, nllloss, calc_entropy, calc_err, l1loss, calc_entropy_log
import pickle
import model
from itertools import combinations
import data_manager as dm
import json
import soundfile
from speech_tools import world_decode_mc, world_speech_synthesis


def load_pickle(path):
    with open(path, 'rb') as f:
        #print(pickle.load(f))
        return pickle.load(f)

def load_sp(feat_dir, num_mcep=36):
    feat_path = os.path.join(feat_dir, 'feats.p')
    with open(feat_path, 'rb') as f:
        sp, _, _, _, _, _= pickle.load(f)
    return sp

def load_phone(feat_dir, num_mcep=36) :
    feat_path = os.path.join(feat_dir, 'feats.p')
    with open(feat_path, 'rb') as f:
        _, _, _, _, _, phone = pickle.load(f)
    return phone

def load_ppg(feat_dir, num_mcep=36):
    ppg_path = os.path.join(feat_dir, 'ppg{}.p'.format(num_mcep))
    with open(ppg_path, 'rb') as f:
        ppg = pickle.load(f)
    return ppg

def calc_parm_num(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def update_parm(opt_list, loss):
    for opt in opt_list:
        opt.zero_grad()
    loss.backward()
    for opt in opt_list:
        opt.step()

def make_one_hot_vector(spk_idx, spk_num):
    vec = np.zeros(spk_num)
    vec[spk_idx] = 1.0
    return vec

def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add(mu)

def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):

    # Logarithm Gaussian normalization for Pitch Conversions
    f0_normalized_t = np.exp((np.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_normalized_t


parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str) # VAE3 MD

parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--model_dir', default='pretrainSI')
parser.add_argument('--lr', type=float, default=1)
#parser.add_argument('--c_lr', type=float, default=2.5*1e-5)
parser.add_argument('--c_lr', type=float, default=0.002)

parser.add_argument('--lr_sch',type=str, default='linear15')
parser.add_argument('--epochs',type=int, default=1000)

parser.add_argument('--baseline',type=str, default='')
parser.add_argument('--disentanglement', type=str, default='')
parser.add_argument('--ws', type=int, default=1)
parser.add_argument('--spk', type=str, default='')

parser.add_argument('--version', type=str, default='')
parser.add_argument('--voted', type=bool, default=False)

args = parser.parse_args()
assert args.model_type in ["VAE1", "VAE2", "VAE3", "MD"]
assert args.version in ['conv', 'mlp', '128', 'final', 'final_mlp']

model_ver = args.version
voted = args.voted

is_MD=True if args.model_type=="MD" else False

torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
np.random.seed(args.seed)

SPK_LIST = ['F10', 'F20', 'F30', 'F40', 'F50', 'M10', 'M20', 'M30', 'M40', 'M50']
TOTAL_SPK_NUM = len(SPK_LIST)
print("TOTAL SPK NUM >> ",TOTAL_SPK_NUM)

SPK_DICT = {
    spk_idx:spk_id 
    for spk_idx, spk_id in enumerate(SPK_LIST)
}
VEC_DICT = {
    spk_id:[make_one_hot_vector(spk_idx, len(SPK_LIST))]
    for spk_idx, spk_id in SPK_DICT.items()
}



# train 데이터의 각 화자의 feats.p에서 sp 값을 추출 
# sp, _, _, _, _ = pickle.load(f)
SP_DICT_TRAIN = {
    spk_id:load_sp(os.path.join("data_seoulNarrative_addLI2_1","train", spk_id)) 
    for spk_id in SPK_LIST
}
SP_DICT_DEV = dict()
for spk_id in SPK_LIST:
    sps = []
    for _, _, file_list in os.walk(os.path.join("data_seoulNarrative_addLI2_1", "dev", spk_id)):
        for file_id in file_list:
            utt_id = file_id.split(".")[0]
            if utt_id == "ppg36":
                continue
            file_path = os.path.join("data_seoulNarrative_addLI2_1", "dev", spk_id, file_id)
            sp, coded_sp, f0, ap, _ = load_pickle(file_path)
            sps.append(coded_sp)
    SP_DICT_DEV[spk_id]=sps
# SP_DICT_TRAIN { speaker : [sp0, sp1, sp2, ... ]}

PHONE_DICT_TRAIN = {
    spk_id:load_phone(os.path.join("data_seoulNarrative_addLI2_1","train", spk_id)) 
    for spk_id in SPK_LIST
}
PHONE_DICT_DEV = dict()
for spk_id in SPK_LIST:
    phones = []
    for _, _, file_list in os.walk(os.path.join("data_seoulNarrative_addLI2_1", "dev", spk_id)):
        for file_id in file_list:
            utt_id = file_id.split(".")[0]
            if utt_id == "ppg36":
                continue
            file_path = os.path.join("data_seoulNarrative_addLI2_1", "dev", spk_id, file_id)
            sp, coded_sp, f0, ap, phone = load_pickle(file_path)
            phones.append(phone)
    PHONE_DICT_DEV[spk_id]=phones


# Model initilaization
model_dir = args.model_dir
os.makedirs(model_dir,exist_ok=True)

latent_dim=8

lr = 1
c_lr = args.c_lr

batch_size = 8

epochs = args.epochs

total_time = 0

min_dev_loss = 9999999999999999
min_epoch = 0
d_epoch = 1

pre_vae = model.VAE(style_dim=TOTAL_SPK_NUM, latent_dim=latent_dim, vae_type=args.model_type, weight_sharing=args.ws )
pre_vae.load_state_dict(torch.load(args.baseline))
pre_vae.cuda()
pre_vae.eval()

if model_ver == "conv" : 
    lang_C = model.LangClassifier(latent_dim=8, label_num=196)
elif model_ver == 'mlp' : 
    lang_C = model.LangClassifier2(latent_dim=8, label_num=60) 
elif model_ver == '128':
    #lang_C = model.LangClassifier128(latent_dim=8, label_num=60) 
    lang_C = model.LangClassifier128(latent_dim=8, label_num=47) 
elif model_ver == 'final':
    lang_C = model.LangClassifierFinal(latent_dim=8, label_num=48)
elif model_ver == 'final_mlp' : 
    lang_C = model.LangClassifierFinal_MLP(latent_dim=8, label_num=48)

lang_C.cuda()
#lang_C_opt = optim.Adam(lang_C.parameters(), lr=c_lr)
lang_C_opt = optim.SGD(lang_C.parameters(), lr=c_lr)
#lang_C_sch = optim.lr_scheduler.LambdaLR(optimizer=lang_C_opt, lr_lambda=lambda epoch: c_lr*(-(1e-2/(args.epochs+2000+1))*epoch+1e-2))
#print("loaded LI model >> ",args.li_path)
print(calc_parm_num(lang_C))
print(lang_C)

torch.save(lang_C.state_dict(), os.path.join(model_dir,"li_{}.pt".format(epochs)))

lm = LogManager()
lm.alloc_stat_type_list(["train_loss", "train_acc", "dev_loss", "dev_acc"])

for epoch in range(epochs+1):
    print("LI Epoch: {}     LearningRate:   {}".format(epoch, lang_C_opt.param_groups[0]['lr']))

    lm.init_stat()  

    lang_C.train()
    train_loader = dm.feat_loader_single3(SP_DICT_TRAIN, batch_size, shuffle=True, ppg_dict=None, phone_dict=PHONE_DICT_TRAIN, voted=False, is_dev=False)      
    
    for self_idx, (coded_mcep, target_phone) in train_loader:
        one_hot_self = dm.make_spk_vector(self_idx, TOTAL_SPK_NUM, batch_size, is_MD)
        
        total_loss = 0.0
        z_mu, z_logvar, z, x_prime_mu, x_prime_logvar, x_prime = pre_vae(x=coded_mcep,one_hot_src=one_hot_self, one_hot_tar=one_hot_self)
        
        self_vec = dm.make_spk_target(self_idx, batch_size, is_MD=False)

        cross_entropy_loss = torch.nn.CrossEntropyLoss()

        if model_ver == "conv" : 
            (li_z, li_z2) = lang_C(z)
            total_li_loss = cross_entropy_loss(li_z, target_phone.unsqueeze(1))
            total_li_err = calc_err(li_z2, target_phone.unsqueeze(1))
        elif model_ver == 'mlp' : 
            li_result = lang_C(z)
            assert len(li_result) == 32, "length of li_result is not 32, {}".format(len(li_result))
            
            total_li_loss = 0
            total_li_err = 0
            # li_result : (32, batch, 196)
            # label     : (batch, 32)
            for i in range(target_phone.shape[-1]) : 
                li_loss = cross_entropy_loss(li_result[i], target_phone[:,i])
                total_li_loss += li_loss
                li_err = torch.mean((torch.argmax(li_result[i], dim=1) != target_phone[:, i]).float())
                total_li_err += li_err
            
            total_li_loss = total_li_loss / target_phone.shape[-1]
            total_li_err = total_li_err / target_phone.shape[-1]
        elif model_ver == '128':
            total_li_loss = 0
            total_li_err = 0
            pi = 0
            for i in range(z.shape[-1]) : #32
                z_ = z[:,:,:,i].unsqueeze(3) # (n, 8, 1, 1)
                (li_z, li_z2) = lang_C(z_)   # (n, 47, 1, 4)
                li_z = li_z.squeeze()   # (n, 47, 4)
                li_z2 = li_z2.squeeze() # (n, 47, 4)
                target = target_phone[:,pi:pi+4] # (n, 128) -> (n, 4)
                pi += 4

                total_li_loss += cross_entropy_loss(li_z, target)
                total_li_err += calc_err(li_z2, target)

            total_li_loss /= z.shape[-1]
            total_li_err /= z.shape[-1]

        elif model_ver == 'final' or model_ver == 'final_mlp':
            (li_z, li_z2) = lang_C(z)   # (n, 47, 1, 32)
            li_z = li_z.squeeze()   # (n, 47, 32)
            li_z2 = li_z2.squeeze() # (n, 47, 32)
            #print(li_z2.shape, target_phone.shape)
            total_li_loss = cross_entropy_loss(li_z, target_phone)
            total_li_err = calc_err(li_z2, target_phone, is_LI=True)
            

        '''
        print("TESTING  TESTING  TESTING  TESTING  TESTING  TESTING  TESTING  TESTING  TESTING  TESTING  TESTING  ")
        print("z : {}".format(z[0]))
        print()
        for i in range((target_phone.shape[0])) : 
            print("li_result : {}".format(torch.argmax(li_result[0], dim=0)))
            print("li_result : {}".format(torch.argmax(li_result[0], dim=1)))
            print("li_result : {}".format(torch.argmax(li_result[0], dim=0)[0]))
            print("li_result : {}".format(torch.argmax(li_result[0], dim=1)[1]))
        print()
        print("label : {}".format(target_phone[:,0]))
        print()
        print("li_err : {}".format(torch.mean((torch.argmax(li_result[0]) != target_phone[:, 0]).float())))

        exit()
        '''


        # 역전파 실행 전 gradient를 0으로 만든다
        lang_C_opt.zero_grad()
        # 역전파 단계 실행
        total_li_loss.backward()
        # 경사하강법 시작 (optimzer는 .grad에 저장된 변화도에 따라 각 매개변수를 조정한다)
        lang_C_opt.step()

        lm.add_torch_stat("train_loss", total_li_loss)
        lm.add_torch_stat("train_acc", 1-total_li_err)

    print("Train:", end=' ')
    lm.print_stat()

    lm.init_stat()
    lang_C.eval()
    dev_loader = dm.feat_loader_single3(SP_DICT_DEV, batch_size, shuffle=True, ppg_dict=None, phone_dict=PHONE_DICT_DEV, voted=False, is_dev=True)      
    for self_idx, (coded_mcep, target_phone) in dev_loader:
        
        one_hot_self = dm.make_spk_vector(self_idx, TOTAL_SPK_NUM, batch_size, is_MD)
        
        total_loss = 0.0
        #print("INPUT!!!!!!!!!!! : {}".format(coded_mcep.shape))
        z_mu, z_logvar, z, x_prime_mu, x_prime_logvar, x_prime = pre_vae(x=coded_mcep,one_hot_src=one_hot_self, one_hot_tar=one_hot_self)
        
        # Latent Classifier
        self_vec = dm.make_spk_target(self_idx, batch_size, is_MD=False)
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        
        if model_ver == "conv" : 
            (li_z, li_z2) = lang_C(z)
            total_li_loss = cross_entropy_loss(li_z, target_phone.unsqueeze(1))
            total_li_err = calc_err(li_z2, target_phone.unsqueeze(1))
        elif model_ver == 'mlp' : 
            li_result = lang_C(z)
            assert len(li_result) == 32, "length of li_result is not 32, {}".format(len(li_result))
            
            total_li_loss = 0
            total_li_err = 0
            # li_result : (32, batch, 196)
            # label     : (batch, 32)
            for i in range(target_phone.shape[-1]) : 
                li_loss = cross_entropy_loss(li_result[i], target_phone[:,i])
                total_li_loss += li_loss
                li_err = torch.mean((torch.argmax(li_result[i], dim=1) != target_phone[:, i]).float())
                total_li_err += li_err
            
            total_li_loss = total_li_loss / target_phone.shape[-1]
            total_li_err = total_li_err / target_phone.shape[-1]
        elif model_ver == '128':
            total_li_loss = 0
            total_li_err = 0
            pi = 0
            #print("TTTTTTTTTTTTTTTTTTT{}".format(target_phone.shape))
            #print(target_phone[0])
            for i in range(z.shape[-1]) : #32
                z_ = z[:,:,:,i].unsqueeze(3) # (n, 8, 1, 1)
                (li_z, li_z2) = lang_C(z_)   # (n, 47, 1, 4)
                li_z = li_z.squeeze()   # (n, 47, 4)
                li_z2 = li_z2.squeeze() # (n, 47, 4)
                target = target_phone[:,pi:pi+4] # (n, 128) -> (n, 4)
                pi += 4

                total_li_loss += cross_entropy_loss(li_z, target)
                total_li_err += calc_err(li_z2, target)

            total_li_loss /= z.shape[-1]
            total_li_err /= z.shape[-1]

        elif model_ver == 'final' :
            (li_z, li_z2) = lang_C(z)   # (n, 47, 1, 32)
            li_z = li_z.squeeze()   # (n, 47, 32)
            li_z2 = li_z2.squeeze() # (n, 47, 32)
            total_li_loss = cross_entropy_loss(li_z, target_phone)
            total_li_err = calc_err(li_z2, target_phone, is_LI=True)
            
    
        lm.add_torch_stat("dev_loss", total_li_loss)
        lm.add_torch_stat("dev_acc", 1-total_li_err)
    
    print("DEV:", end=' ')
    lm.print_stat()
    print(".....................")
    # lang_C_sch.step()
    if epoch % 10 == 0 : 
        torch.save(lang_C.state_dict(), os.path.join(model_dir,"li_{}.pt".format(epoch)))


torch.save(lang_C.state_dict(), os.path.join(model_dir,"li_{}.pt".format(epochs)))

import os
import glob
import librosa
import subprocess
import time
import pickle
import torch.nn.functional as F

train_dir = '../corpus_seoulNarrative/train_script/'
dev_dir = '../corpus_seoulNarrative/dev_script/'
os.makedirs('../corpus_seoulNarrative/dev',exist_ok=True)
os.makedirs('../corpus_seoulNarrative/dev_script',exist_ok=True)

spk_list = ['M1','F2','M2']
# spk_list = ['F10', 'F20', 'F30', 'F40', 'M10', 'M20', 'M30', 'M40', 'F50', 'M50']

for spk in spk_list:
    os.makedirs(os.path.join(dev_dir,spk),exist_ok=True)

'''
train_f1 = glob.glob('../corpus_seoulNarrative/train/F10/*.wav')
train_f2 = glob.glob('../corpus_seoulNarrative/train/F20/*.wav')
train_f3 = glob.glob('../corpus_seoulNarrative/train/F30/*.wav')
train_f4 = glob.glob('../corpus_seoulNarrative/train/F40/*.wav')
train_m1 = glob.glob('../corpus_seoulNarrative/train/M10/*.wav')
train_m2 = glob.glob('../corpus_seoulNarrative/train/M20/*.wav')
train_m3 = glob.glob('../corpus_seoulNarrative/train/M30/*.wav')
train_m4 = glob.glob('../corpus_seoulNarrative/train/M40/*.wav')
train_f5 = glob.glob('../corpus_seoulNarrative/train/F50/*.wav')
train_m5 = glob.glob('../corpus_seoulNarrative/train/M50/*.wav')

train_set = {'F10':train_f1, 'M10':train_m1, 'F20':train_f2, 'M20':train_m2, 'F30':train_f3, 'M30':train_m3, 'F40':train_f4, 'M40':train_m4, 'F50':train_f1, 'M50':train_m1}

train_m1_set = set([f.split('/')[-1] for f in train_m1])
train_m2_set = set([f.split('/')[-1] for f in train_m2])
train_m3_set = set([f.split('/')[-1] for f in train_m3])
train_m4_set = set([f.split('/')[-1] for f in train_m4])
train_f1_set = set([f.split('/')[-1] for f in train_f1])
train_f2_set = set([f.split('/')[-1] for f in train_f2])
train_f3_set = set([f.split('/')[-1] for f in train_f3])
train_f4_set = set([f.split('/')[-1] for f in train_f4])
train_f5_set = set([f.split('/')[-1] for f in train_f1])
train_m5_set = set([f.split('/')[-1] for f in train_m1])

intersection = train_m1_set&train_m2_set&train_m3_set&train_m4_set&train_f1_set&train_f2_set&train_f3_set&train_f4_set&train_m5_set&train_f5_set
'''
train_f1 = glob.glob('../corpus_seoulNarrative/train/F1/*.wav')
train_f2 = glob.glob('../corpus_seoulNarrative/train/F2/*.wav')
train_m1 = glob.glob('../corpus_seoulNarrative/train/M1/*.wav')
train_m2 = glob.glob('../corpus_seoulNarrative/train/M2/*.wav')
train_f1_set = set([f.split('/')[-1] for f in train_f1])
train_f2_set = set([f.split('/')[-1] for f in train_f2])
train_m1_set = set([f.split('/')[-1] for f in train_m1])
train_m2_set = set([f.split('/')[-1] for f in train_m2])
intersection = train_m1_set&train_m2_set&train_f1_set&train_f2_set

dev_list = list(intersection)[:40]

print(dev_list)

print(len(glob.glob('../corpus_seoulNarrative/train/F50/*.wav')))

exp_dir = os.path.join('processed')


sampling_rate = 16000
num_mcep = 36
frame_period = 5.0
n_frames = 128


dataset_loc = '../data_seoulNarrative_addLI'

total_length = 0
total_data_n = 0
min_len = 999999
max_len = -1

phones = ["G1","GG1","N1","D1","DD1","R1","M1","B1","BB1","S1","SS1","O1","J1","JJ1","C1","K1","T1","P1",
"H1","A2","AE2","YA2","YAE2","EO2","E2","YEO2","YE2","O2","WA2","WAE2","OE2","YO2","U2","WEO2","WE2","WI2","YU2","EU2","YI2","I2",
"G3","GG3","GS3","N3","NJ3","NH3","D3","L3","LG3","LM3","LB3","LS3","LT3","LP3","LH3","M3","B3","BS3","NG3","W3"]
phone_dict = {}
for i in range(len(phones)) : 
    phone_dict[phones[i]] = i

print("phone list length : {}".format(len(phones)))

for speaker in spk_list:
    train_sps = []
    train_f0s = []
    train_hmm = []
    for dtype in ["train", "test"]:
        corpus_dir = os.path.join('../corpus_seoulNarrative',dtype,speaker)
        target_dir = os.path.join('../corpus_seoulNarrative', dtype + '_phone', speaker)
        os.makedirs(target_dir, exist_ok=True)

        non_train_dict = dict()

        print('Loading {} Wavs...'.format(speaker))
        f = glob.glob(corpus_dir+'/*')

        for path in f:
            info = path.split(".")[-2]
            utt_id = info.split("/")[-1]                # /1_s03 같이 확장자 없는 파일 이름
            script_id = '_'.join(utt_id.split('_')[1:]) # s03 이 부분

            script_path = os.path.join('/home/klklp98/speechst2/Exp_Disentanglement/corpus_seoulNarrative', dtype+'_script', speaker, os.path.basename(path).split('.')[0]+'.txt')
            test_path = os.path.join('/home/klklp98/speechst2/Exp_Disentanglement/preprocess/t01_s04.txt')
            print("Processing",utt_id)
            
            if os.path.isfile("/home/klklp98/speechst2/Exp_Disentanglement/preprocess/pronunciation/converted.txt") : 
                os.remove("/home/klklp98/speechst2/Exp_Disentanglement/preprocess/pronunciation/converted.txt")
            time.sleep(1)


            #pronunciated = subprocess.call(["chmod", "4755", "./pronunciation.sh", test_path])
            os.system("Pronunciation -ifmt UNICODE8 -ofmt ROMANIZEDHANGUL -log ./pronunciation/log.txt -logfmt UNICODE8 -use_syllable T {} ./pronunciation/converted.txt".format(script_path))

            time.sleep(1)
            
            with open("/home/klklp98/speechst2/Exp_Disentanglement/preprocess/pronunciation/converted.txt", 'r') as phones :
                tmp_list = phones.read().split("\n")
            
            for idx, i in enumerate(tmp_list) : 
                tmp_list[idx] = tmp_list[idx].split('\t')[-1]

            # tmp_list = [ "B1O2W3 M1I2W3 M1YEO2N3", "BB1EO2W3 GG1U2W3 G1I2W3" ]
            phone_list = []
            for idx, i in enumerate(tmp_list) :
                l = tmp_list[idx]
                l = l.replace(' ', '')
                # print(l)
                # l = ""B1O2W3M1I2W3M1YEO2N3""
                buf = []
                for j in range(len(l)) :
                    if l[j].isdigit() == False : 
                        buf.append(l[j])
                    elif l[j] == ' ' : 
                        continue
                    else :
                        buf.append(l[j])
                        phone_list.append(phone_dict[''.join(buf)])
                        buf = []

            phone_list = list(map(int, phone_list))
            # print(phone_list)
            # exit()
            length = len(phone_list)
            total_length += length
            total_data_n += 1
            if min_len > length :
                min_len = length
            if max_len < length : 
                max_len = length

            p_dir = os.path.join(target_dir, utt_id+'.p')
            with open(p_dir, "wb") as p : 
                pickle.dump(phone_list, p)


print("total length : ", total_length)
print("mean length : ", (total_length / total_data_n))
print("max length : ", max_len)
print("min length : ", min_len)
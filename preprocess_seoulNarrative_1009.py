import os, sys
import time
import gc
import torch
import soundfile
import librosa
import glob
import shutil
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from speech_tools import world_decode_mc, world_speech_synthesis
from preprocess_tools import *


train_dir = '../corpus_seoulNarrative/train/'
dev_dir = '../corpus_seoulNarrative/dev/'
dev_flac_dir = '../corpus_seoulNarrative/dev_flac/'
dev_phone_dir = '../corpus_seoulNarrative/dev_phone/'
os.makedirs('../corpus_seoulNarrative/dev',exist_ok=True)
os.makedirs('../corpus_seoulNarrative/dev_flac',exist_ok=True)
os.makedirs('../corpus_seoulNarrative/dev_phone',exist_ok=True)

def make_one_hot_vector(spk_idx, spk_num):
    vec = np.zeros(spk_num)
    vec[spk_idx] = 1.0
    return vec

def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):
    
    # Logarithm Gaussian normalization for Pitch Conversions
    f0_normalized_t = np.exp((np.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_normalized_t

#spk_list = ['F1','M1','F2','M2']
spk_list = ['F1','M1','F2','M2','F10', 'M10', 'F20', 'M20', 'F30', 'M30', 'F40', 'M40', 'F50', 'M50']
TOTAL_SPK_NUM = len(spk_list)
print("TOTAL SPK NUM >> ",TOTAL_SPK_NUM)

SPK_DICT = {
    spk_idx:spk_id 
    for spk_idx, spk_id in enumerate(spk_list)
}
VEC_DICT = {
    spk_id:[make_one_hot_vector(spk_idx, len(spk_list))]
    for spk_idx, spk_id in SPK_DICT.items()
}

for spk in spk_list:
    os.makedirs(os.path.join(dev_dir,spk),exist_ok=True)

train_f1 = glob.glob('../corpus_seoulNarrative/train/F10/*.wav')
train_f2 = glob.glob('../corpus_seoulNarrative/train/F20/*.wav')
train_f3 = glob.glob('../corpus_seoulNarrative/train/F30/*.wav')
train_f4 = glob.glob('../corpus_seoulNarrative/train/F40/*.wav')
train_f5 = glob.glob('../corpus_seoulNarrative/train/F1/*.wav')
train_f6 = glob.glob('../corpus_seoulNarrative/train/F2/*.wav')
train_f7 = glob.glob('../corpus_seoulNarrative/train/F50/*.wav')
train_m1 = glob.glob('../corpus_seoulNarrative/train/M10/*.wav')
train_m2 = glob.glob('../corpus_seoulNarrative/train/M20/*.wav')
train_m3 = glob.glob('../corpus_seoulNarrative/train/M30/*.wav')
train_m4 = glob.glob('../corpus_seoulNarrative/train/M40/*.wav')
train_m5 = glob.glob('../corpus_seoulNarrative/train/M1/*.wav')
train_m6 = glob.glob('../corpus_seoulNarrative/train/M2/*.wav')
train_m7 = glob.glob('../corpus_seoulNarrative/train/M50/*.wav')

train_set = {'F1':train_f5, 'M1':train_m5, 'F2':train_f6, 'M2':train_m6, 'F10':train_f1, 'M10':train_m1, 'F20':train_f2, 'M20':train_m2, 'F30':train_f3, 'M30':train_m3, 'F40':train_f4, 'M40':train_m4, 'F50':train_f7, 'M50':train_m7}

train_m1_set = set([f.split('/')[-1] for f in train_m1])
train_m2_set = set([f.split('/')[-1] for f in train_m2])
train_m3_set = set([f.split('/')[-1] for f in train_m3])
train_m4_set = set([f.split('/')[-1] for f in train_m4])
train_m5_set = set([f.split('/')[-1] for f in train_m5])
train_m6_set = set([f.split('/')[-1] for f in train_m6])
train_m7_set = set([f.split('/')[-1] for f in train_m7])
train_f1_set = set([f.split('/')[-1] for f in train_f1])
train_f2_set = set([f.split('/')[-1] for f in train_f2])
train_f3_set = set([f.split('/')[-1] for f in train_f3])
train_f4_set = set([f.split('/')[-1] for f in train_f4])
train_f5_set = set([f.split('/')[-1] for f in train_f5])
train_f6_set = set([f.split('/')[-1] for f in train_f6])
train_f7_set = set([f.split('/')[-1] for f in train_f7])

intersection = train_m1_set&train_m2_set&train_m3_set&train_m4_set&train_m5_set&train_m6_set&train_m7_set&train_f1_set&train_f2_set&train_f3_set&train_f4_set&train_f5_set&train_f6_set&train_f7_set

dev_list = list(intersection)[:40]


print(dev_list)

print(len(glob.glob('../corpus_seoulNarrative/train/F50/*.wav')))

exp_dir = os.path.join('processed')

sampling_rate = 16000
num_mcep = 36
frame_period = 5.0
n_frames = 128

# train/VCC2SF1 =

dataset_loc = '../data_seoulNarrative_addLI2_1'

ctm_path = "/home/klklp98/speechst2/zeroth_n/zeroth/s5/EXTRACT/phones.ctm"
flac_path = "/home/klklp98/speechst2/Exp_Disentanglement/for_LI/phone_check.flac"
frame_shift = 0.01
SIL_set = set([1, 2, 3, 4, 5])
SPN_set = set([0, 6, 7, 8, 9])

max_err_frame = -1

for speaker in spk_list:
    train_sps = []
    f0s_list = []
    sps_ori_list = []
    aps_list = []
    train_phone = []
    for dtype in ["train", "test"]:
        corpus_dir = os.path.join('../corpus_seoulNarrative',dtype,speaker)
        data_dir  = os.path.join(dataset_loc,dtype,speaker)

        non_train_dict = dict()

        # train, dev: feats.p, ppgs.p
        # test: feats.p

        os.makedirs(data_dir, exist_ok=True)
        if dtype == "train":
            dev_dir = os.path.join(dataset_loc, 'dev', speaker)
            os.makedirs(dev_dir, exist_ok=True)

        print('Loading {} Wavs...'.format(speaker))
        f = glob.glob(corpus_dir+'/*')

        lnl = 0
        for path in f:
            # path = path[:-1]
            info = path.split(".")[-2]
            utt_id = info.split("/")[-1]

            script_id = '_'.join(utt_id.split('_')[1:])

            print("Processing",utt_id)
            
            wav, _ = librosa.load(path, sr = sampling_rate, mono = True)
            f0, timeaxis, sp, ap, coded_sp = world_decompose(wav=wav, fs=sampling_rate, frame_period=frame_period, num_mcep=num_mcep)
            frame_num = 4 * (len(f0) // 4)
            coded_sp = coded_sp[:frame_num] # 1556, 36
            f0 = f0[:frame_num]
            ap = ap[:frame_num]
            sp = sp[:frame_num]             # 1556, 513

            soundfile.write(os.path.join("../for_LI", "phone_check.flac"), wav, 16000)
            time.sleep(1)
            subprocess.call(["../for_LI/LI_decode.sh", flac_path], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            time.sleep(1)
            with open(ctm_path, "r") as ctm : 
                cur_phone = []
                for l in ctm.readlines() :
                    num_of_frames = int(float(l.split(' ')[3]) / frame_shift)
                    phone = l.split(' ')[4]

                    # prefix를 제거하고 phone_list 추가
                    if int(phone) in SIL_set : 
                        for i in range(num_of_frames) : 
                            cur_phone.append(0)
                    elif int(phone) in SPN_set : 
                        assert True == False , "There is SPN phone in {}".format(l)
                        #minus_list.append(l)
                    else : 
                        phone = int(phone)
                        phone -= 10
                        phone = (phone // 4) + 1
                        if phone < 0 :
                            assert True == False , "phone is minus"
                        for i in range(num_of_frames) : 
                            cur_phone.append(phone)

            if len(cur_phone) < (frame_num//4) :
                left = (frame_num//4) - len(cur_phone)
                last = cur_phone[-1]
                for i in range(left) :
                    cur_phone.append(last)
            elif len(cur_phone) > (frame_num//4) : 
                left = len(cur_phone) - (frame_num//4)
                cur_phone = cur_phone[:(frame_num//4)]
            else :
                left = 0
            if left > max_err_frame : 
                    max_err_frame = left

            lnl += 1
            if dtype=="train" and utt_id+'.wav' not in dev_list:
                # train
                # 각 파일들을 append 한다
                f0s_list.append(f0)
                sps_ori_list.append(sp)
                aps_list.append(ap)
                train_sps.append(coded_sp.T)
                train_phone.append(cur_phone)
                print("--------------------------")
                print("[ {} ]".format(utt_id))
                print("\twav frame : {}".format(frame_num))
                print("\tphone frame : {}".format(len(cur_phone)))

                #assert len(cur_phone) <= (frame_num//4) + 3 and len(cur_phone) >= (frame_num//4) - 3, "phone len is too different bb"
                
            else:
                print("     is {} >> ".format('dev' if dtype=='train' else dtype),utt_id)

                non_train_dict[utt_id] = (sp, coded_sp.T, f0, ap, cur_phone)
                print("--------------------------")
                print("[ {} ]".format(utt_id))
                print("\twav frame : {}".format(frame_num))
                print("\tphone frame : {}".format(len(cur_phone)))
                #assert len(cur_phone) <= (frame_num//4) + 3 and len(cur_phone) >= (frame_num//4) - 3, "phone len is too different bb"


        print('Saving {} data to {}... '.format(speaker, dtype))
        if dtype=='train':
            # f0s_list => (utt개수, 36, 각 utt의 프레임 수)
            log_f0s_mean, log_f0s_std = logf0_statistics(f0s_list)
            train_sps_norm, sps_mean, sps_std = mcs_normalization_fit_transform(mcs=train_sps)

            assert len(train_sps_norm) == len(f0s_list) == len(sps_ori_list) == len(aps_list), "The lens are different!"
            #assert train_sps_norm[0].T.shape == sps_ori_list[0].shape == f0s_list[0].shape == aps_list[0].shape, "The shapes are different! {} {} {} {}".format(train_sps_norm[0].T.shape, sps_ori_list[0].shape, f0s_list[0].shape, aps_list[0].shape)

            save_pickle(os.path.join(dataset_loc, 'train', speaker, 'feats.p'),
                        (train_sps_norm, sps_mean, sps_std, log_f0s_mean, log_f0s_std, train_phone))

            data_dir = os.path.join(dataset_loc, 'dev', speaker)
            corpus_dir_dev = os.path.join('../corpus_seoulNarrative','dev',speaker)
            
            for utt_id, (sp, coded_sp, f0, ap, cur_phone) in non_train_dict.items():
                # print("Processing dev >> ",utt_id)
                new_sp = (coded_sp-sps_mean) / sps_std
                save_pickle(os.path.join(data_dir, '{}.p'.format(utt_id)), (sp, new_sp, f0, ap, cur_phone))

                shutil.move(os.path.join(corpus_dir,utt_id+'.wav'), os.path.join(corpus_dir_dev,utt_id+'.wav'))
                #shutil.move(os.path.join('../corpus_seoulNarrative', dtype+'_flac', speaker, utt_id+'.flac'), os.path.join(corpus_dir_flac_dev, utt_id+'.flac'))
                print("     moved dev {}.wav ? >> ".format(utt_id), os.path.isfile(os.path.join(corpus_dir_dev,utt_id+'.wav')))

        else:
            data_dir = os.path.join(dataset_loc, dtype, speaker)
            for utt_id, (sp, coded_sp, f0, ap, cur_phone) in non_train_dict.items():
                print("Processing",utt_id)
                new_sp = (coded_sp-sps_mean) / sps_std
                save_pickle(os.path.join(data_dir, '{}.p'.format(utt_id)), (sp, new_sp, f0, ap, cur_phone))

        print("{} has {} files".format(speaker, lnl))

    print("Maximum Frame Err : {}".format(max_err_frame))    
    print('Preprocessing Done.')


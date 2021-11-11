# G_ML-AI_LAB_LIModule

고려대학교 정보대학 인공지능연구실에서 근무하면서 연구한 내용 중 하나입니다.

VAE3의 encoder가 출력하는 latent vector에서 최대한 음소 정보가 살아있을 수 있도록 명시적인 학습 경로를 제공하도록 만든 모듈입니다.

LI모듈은 latent vector로부터 음소를 예측한 후 target 음소와 비교하며 발생하는 loss를 전체 VAE3의 loss function에 추가함으로써
VAE3의 Encoder가 latent vector에 음소 정보를 더 많이 남길 수 있도록 유도합니다.

target 음소는 kaldi library를 통해 모든 train/dev/test 음성 데이터로부터 추출하였으며
그 과정은 LI_decode.sh 를 보시면 알 수 있습니다.

kaldi에 대한 훈련은 
zeroth library와 VAE3의 훈련에 사용된 한국어 데이터를 이용하여 직접 훈련을 진행했습니다.

import easydict

from dataloader import *
from train import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = easydict.EasyDict({
    'num_epochs': 50,
    'num_epochs_ae': 50,
    'lr': 1e-3,
    'lr_ae': 1e-3,
    'weight_decay': 5e-7,
    'weight_decay_ae': 5e-3,
    'lr_milestones': [50],
    'batch_size': 1024,
    'pretrain': True,
    'latent_dim': 32,
    'normal_class': 0
})

if __name__ == '__main__':

    # Train/Test Loader 불러오기
    dataloader_train, dataloader_test = get_mnist(args)

    # Network 학습준비, 구조 불러오기
    deep_SVDD = TrainerDeepSVDD(args, dataloader_train, device)

    # DeepSVDD를 위한 DeepLearning pretrain 모델로 Weight 학습
    if args.pretrain:
        deep_SVDD.pretrain()

    # 학습된 가중치로 Deep_SVDD모델 Train
    net, c = deep_SVDD.train()

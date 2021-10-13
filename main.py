import argparse

from dataloader import *
from train import *
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

writer = SummaryWriter('./runs/experiment1')

parser = argparse.ArgumentParser(description='Train Deep SVDD model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--num_epochs', '-e', type=int, default=50, help='Num of epochs to Deep SVDD train')
parser.add_argument('--num_epochs_ae', '-ea', type=int, default=50, help='Num of epochs to AE model train')
parser.add_argument('--lr', '-lr', type=float, default=1e-3, help='learning rate for model')
parser.add_argument('--lr_ae', '-lr_ae', type=float, default=1e-3, help='learning rate for AE model')
parser.add_argument('--weight_decay', '-wd', type=float, default=5e-7, help='weight decay for model')
parser.add_argument('--weight_decay_ae', '-wd_ae', type=float, default=5e-3, help='weight decay for model')
parser.add_argument('--lr_milestones', '-lr_mile', type=list, default=[50], help='learning rate milestones')
parser.add_argument('--batch_size', '-bs', type=int, default=1024, help='batch size')
parser.add_argument('--pretrain', '-pt', type=bool, default=True, help='Pretrain to AE model')
parser.add_argument('--latent_dim', '-ld', type=int, default=32, help='latent dimension')
parser.add_argument('--normal_class', '-cls', type=int, default=0, help='Set the normal class')

args = parser.parse_args()

if __name__ == '__main__':
    dataloader_train, dataloader_test = get_mnist(args)

    deep_SVDD = TrainerDeepSVDD(args=args, data_loader=dataloader_train, device=device, R=0.0, nu=0.1, writer=writer)

    if args.pretrain:
        print("Start AUTOENCODER train!")
        deep_SVDD.pretrain()

    print("Start Deep SVDD train!")
    net, c = deep_SVDD.train()

    test_auroc = deep_SVDD.test(net=net, test_loader=dataloader_test)
    print("Test AUROC: {:.2f}".format(test_auroc * 100))
    writer.flush()
    writer.close()

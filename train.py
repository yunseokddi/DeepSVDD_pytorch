import torch

from model import pretrain_autoencoder, DeepSVDDNetwork
from tqdm import tqdm


class TrainerDeepSVDD(object):
    def __init__(self, args, data_loader, device):
        self.args = args
        self.train_loader = data_loader
        self.device = device

    def pretrain(self):
        ae = pretrain_autoencoder(self.args.latent_dim).to(self.device)
        ae.apply(weights_init_normal)
        optimizer = torch.optim.Adam(ae.parameters(), lr=self.args.lr_ae,
                                     weight_decay=self.args.weight_decay_ae)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.args.lr_milestones, gamma=0.1)
        ae.train()

        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            tq = tqdm(self.train_loader, total=len(self.train_loader))
            
            for x, _ in tq:
                x = x.float().to(self.device)

                optimizer.zero_grad()
                x_hat = ae(x)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()

                total_loss += reconst_loss.item()
                errors = {
                    'epoch': epoch,
                    'train loss': reconst_loss.item()
                }

                tq.set_postfix(errors)

        scheduler.step()
        print('total_loss: {:.2f}'.format(total_loss))

        self.save_weights_for_DeepSVDD(ae, self.train_loader)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

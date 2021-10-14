import torch
import numpy as np

from model import pretrain_autoencoder, DeepSVDDNetwork
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


class TrainerDeepSVDD(object):
    def __init__(self, args, data_loader, device, R, nu, writer):
        self.args = args
        self.train_loader = data_loader
        self.device = device
        self.R = torch.tensor(R, device=self.device)
        self.nu = nu
        self.warm_up_n_epochs = 10
        self.writer = writer

    def pretrain(self):
        """Pretrain AUTO ENCODER for using Deep SVDD"""
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

            epoch_loss = total_loss / len(self.train_loader)

            self.writer.add_scalar("AE/Loss", epoch_loss, epoch)

        scheduler.step()

        self.save_weights_for_DeepSVDD(ae, self.train_loader)

    def save_weights_for_DeepSVDD(self, model, dataloader):
        """Initializing for Deep SVDD's weights from pretrained AUTO ENCODER's weights"""
        c = self.set_c(model, dataloader)
        net = DeepSVDDNetwork(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()}, './weights/pretrained_weights.pth')

    def set_c(self, model, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        model.eval()
        z_ = []

        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encoder(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)

        # If c is close to 0, set to +-eps
        # To avoid trivial problem
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def train(self):
        """Train the Deep SVDD"""
        net = DeepSVDDNetwork().to(device=self.device)

        if self.args.pretrain == True:
            state_dict = torch.load('./weights/pretrained_weights.pth')
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            net.apply(weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)

        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr,
                                     weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.args.lr_milestones, gamma=0.1)

        net.train()

        for epoch in range(self.args.num_epochs):
            total_loss = 0
            tq = tqdm(self.train_loader, total=len(self.train_loader))

            for x, _ in tq:
                x = x.float().to(self.device)

                optimizer.zero_grad()
                output = net(x)
                dist = torch.sum((output - c) ** 2, dim=1)

                scores = dist - self.R ** 2
                loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                loss.backward()
                optimizer.step()

                if epoch >= self.warm_up_n_epochs:
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                total_loss += loss.item()

                errors = {
                    'epoch': epoch,
                    'train loss': loss.item()
                }

                tq.set_postfix(errors)

            epoch_loss = total_loss / len(self.train_loader)

            self.writer.add_scalar("Deep SVDD/Loss", epoch_loss, epoch)

            scheduler.step()

        torch.save(net.state_dict(), './weights/best_weight.pt')
        self.net = net
        self.c = c

        return self.net, self.c

    def test(self, net, test_loader):
        net.to(self.device)
        net.eval()

        print("Start testing")

        label_score = []
        with torch.no_grad():
            tq = tqdm(test_loader, total=len(test_loader))
            for x, y in tq:
                x = x.to(self.device)
                z = net(x)
                dist = torch.sum((z - self.c) ** 2, dim=1)

                scores = dist - self.R ** 2

                label_score += list(zip(
                    y.cpu().data.numpy().tolist(),
                    scores.cpu().data.numpy().tolist()))

        self.test_scores = label_score
        labels, scores = zip(*label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        self.test_auc = roc_auc_score(labels, scores)

        return self.test_auc


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

import torch
from torch import nn
import numpy as np
from data.dataloader_net import load_train_test


def init_weights(l):
        """The function intializes the weights of linear layers of a network
            using He initialization.
         """
        if isinstance(l, nn.Linear):
            torch.nn.init.kaiming_normal_(l.weight, mode='fan_in', nonlinearity='relu')


class MLPNet(nn.Module):
    def __init__(self, sh_degree_envlight: int = 4, sh_degree_sky: int = 1, embedding_dim: int = 32, dense_layer_size: int = 256):
        super().__init__()
        self.sh_dim_envlight = (sh_degree_envlight + 1) ** 2
        self.sh_dim_sky = (sh_degree_sky + 1) ** 2
        self.embedding_dim = embedding_dim
        self.dense_layer_size = dense_layer_size
        self.optimizer = torch.optim.Adam


        self.base = nn.Sequential(
                nn.Linear(self.embedding_dim, self.dense_layer_size),
                nn.Dropout(p=0.2),
                nn.ReLU(), 
                nn.Linear(self.dense_layer_size, self.dense_layer_size),
                nn.ReLU(),
                nn.Linear(self.dense_layer_size, self.dense_layer_size // 2),
                nn.ReLU(),
            )


        self.sh_sky_outlayer = nn.Linear(self.dense_layer_size // 2, self.sh_dim_sky*3)

        self.sh_envlight_layers = nn.Sequential(nn.Linear(self.dense_layer_size // 2, self.dense_layer_size // 2),
                                            nn.ReLU())
        self.sh_envlight_outlayer = nn.Linear(self.dense_layer_size // 2, self.sh_dim_envlight*3)


    def forward(self, e):
        base_features = self.base(e)
        
        sh_sky = self.sh_sky_outlayer(base_features).view(-1, self.sh_dim_sky, 3)

        sh_envlight = self.sh_envlight_layers(base_features)
        sh_envlight = self.sh_envlight_outlayer(sh_envlight).view(-1, self.sh_dim_envlight, 3)       

        return sh_envlight, sh_sky



    def get_optimizer(self):
        return self.optimizer(self.parameters(), lr=0.002)
    

    def save_weights(self, path: str, epoch: int):
        torch.save(self.state_dict(), path + "/MLPNet_epoch_"+str(epoch)+".pth")


    def initialize_sh_envl(self, dataloader, epochs, optim=None):
        if optim is None:
            optim = self.get_optimizer()
        loss_ = torch.nn.MSELoss()
        for _ in range(epochs):
            self.train()
            for batch in dataloader:
                input = batch[0].squeeze().cuda()
                sh_target = batch[1].cuda()
                sh_out, _ = self(input)
                mse = loss_(sh_target, sh_out)
                mse.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                optim.step()
                optim.zero_grad()


class EmbeddingNet(nn.Module):
    """
    The network is a convolutional autoencoder.
    """
    def __init__(self, latent_dim: int = 32, kernel_size: int = 3,
                 channels_f: int = 128,
                 dense_layer_size: int = 64, input_shape: int = 256):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels_f = channels_f
        self.dense_layer_size = dense_layer_size
        self.latent_dim = latent_dim
        self.ae_loss = torch.nn.MSELoss
        self.input_shape = input_shape
        enc_end_shape = self.channels_f*(np.int32(input_shape/4)**2)
        self.optimizer = torch.optim.Adam
    
        self.encoder_conv = nn.Sequential(nn.Conv2d(3, np.int32(self.channels_f/2), self.kernel_size, padding='same'), # 3x256x256 -> 64x256x256
                                     nn.BatchNorm2d(np.int32(self.channels_f/2)),
                                     nn.ReLU(),
                                     nn.Conv2d(np.int32(self.channels_f/2), np.int32(self.channels_f/2), self.kernel_size, padding='same'),
                                     nn.BatchNorm2d(np.int32(self.channels_f/2)),
                                     nn.ReLU(),
                                     nn.AvgPool2d(kernel_size=2), #64x128x128
                                     nn.Conv2d(np.int32(self.channels_f/2), self.channels_f,  self.kernel_size, padding='same'),
                                     nn.BatchNorm2d(self.channels_f),
                                     nn.ReLU(),
                                     nn.Conv2d(self.channels_f, self.channels_f,  self.kernel_size, padding='same'), #128x128x128
                                     nn.BatchNorm2d(self.channels_f),
                                     nn.ReLU(),
                                     nn.AvgPool2d(kernel_size=2) # 128x64x64
                                    )
        self.encoder_dense = nn.Linear(enc_end_shape, self.latent_dim)

        self.decoder_conv = nn.Sequential(
                                     nn.ConvTranspose2d(self.channels_f, self.channels_f,  self.kernel_size, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(self.channels_f),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(self.channels_f, np.int32(self.channels_f/2), self.kernel_size, padding=1),
                                     nn.BatchNorm2d(np.int32(self.channels_f/2)),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(np.int32(self.channels_f/2), np.int32(self.channels_f/2),  self.kernel_size, stride=2, padding=1, output_padding=1),
                                     nn.BatchNorm2d(np.int32(self.channels_f/2)),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(np.int32(self.channels_f/2), 3, self.kernel_size, padding=1),
                                     nn.BatchNorm2d(3),
                                     nn.ReLU()
                                     )
        self.decoder_dense = nn.Linear(self.latent_dim, enc_end_shape)

        for linear_layer in [self.encoder_dense, self.decoder_dense]:
             linear_layer.apply(init_weights)


    def forward(self, x, pretraining=False):
         x_conv = self.encoder_conv(x)
         feature_vec = self.encoder_dense(torch.flatten(x_conv, start_dim=1))
         if pretraining:
            decoded = self.decoder_dense(feature_vec)
            decoded = torch.unflatten(decoded, dim=1, sizes=(self.channels_f, np.int32(self.input_shape/4), np.int32(self.input_shape/4)))
            decoded = self.decoder_conv(decoded)
            return decoded
         else:
             return feature_vec


    def get_optimizer(self):
        return self.optimizer(self.parameters(), lr=0.001, weight_decay=1e-5)


    def save_weights(self, path: str, epoch: int):
        torch.save(self.state_dict(), path + "/EmbeddingNet_model_epoch_"+str(epoch)+".pth")

   
    def optimize(self, data_path, num_epochs: int = 50, resize_dim: int = 256, batch_size: int = 32,
                    verbose=False, tensorboard_writer=None,
                    progress_bar=None, output_path=None, get_datatransforms_only = False, 
                    return_outputs=False):
        if get_datatransforms_only:
            _, _, data_transforms = load_train_test(datapath=data_path, resize_dim=resize_dim, batch_size=batch_size)
            return data_transforms
        self.cuda()
        loss_ = self.ae_loss()
        losses_tr_all = []
        losses_test_all = []
        optim = self.get_optimizer()

        for epoch in range(num_epochs):
            self.train()
            losses_tr = []
            train_iter, test_iter, data_transforms = load_train_test(datapath=data_path, resize_dim=resize_dim, batch_size=batch_size)

            for batch in train_iter:
                optim.zero_grad() 
                input = batch.cuda()
                reconstructed_input = self(input, pretraining=True)
                mse = loss_(reconstructed_input, input)
                mse.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                optim.step() 
                losses_tr.append(mse.detach().cpu().item())
            losses_tr_all.append(np.mean(losses_tr))
            if (tensorboard_writer):
                tensorboard_writer.add_scalar('train loss- ae',
                                losses_tr_all[epoch],
                                epoch)

            self.eval()
            losses_test = []
            with torch.no_grad():
                for batch in test_iter:
                    input = batch.cuda()
                    reconstructed_input = self(input, pretraining=True)
                    mse = loss_(reconstructed_input, input)
                    losses_test.append(mse.detach().cpu().item())
            torch.cuda.empty_cache()
            losses_test_all.append(np.mean(losses_test))
            if tensorboard_writer:
                tensorboard_writer.add_scalar('test loss- ae',
                            losses_test_all[epoch],
                            epoch)
            if verbose:
                print(f"Epoch: {epoch}, train loss: {losses_tr_all[epoch]}, test_loss: {losses_test_all[epoch]}")
            elif progress_bar:
                if epoch % 1 == 0:
                    progress_bar.set_postfix({"Loss test": f"{losses_test_all[epoch]:.{4}f}"})
                    progress_bar.update(1)
                if epoch == num_epochs -1:
                    progress_bar.close()

        if output_path:
            self.save_weights(output_path, epoch)
        
        if return_outputs:
            out = {'test_losses': losses_test_all, 
                   'train_losses': losses_tr_all,
                   'data_transforms': data_transforms
                   }
            return out




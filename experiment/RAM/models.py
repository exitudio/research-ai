import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from helpers import append, resize, device


class GlimpseSensor:
    def __init__(self, retina_size, num_zoom_image=3, scale=2):
        """
        Args
        ----
        retina_size - smallest size of image
        """
        self.retina_size = retina_size
        self.num_zoom_image = num_zoom_image
        self.scale = scale

    def foveate(self, imgs, locations):
        """
        Args
        ----
        imgs - (B, C, H, W)
        location - (B, H, W) [-1, 1]
        """
        glimpse_imgs = None
        image_size = imgs[0].shape[-1]
        denomalized_locations = (locations+1) * .5 * image_size
        for i, img in enumerate(imgs):
            img = self._glimp_patch(img, denomalized_locations[i], image_size)
            glimpse_imgs = append(glimpse_imgs, img)
        return glimpse_imgs.view(glimpse_imgs.shape[0], -1)  # flatten

    def _glimp_patch(self, img, location, image_size):
        # TODO it is an inaccurate if image size is odd number.
        glimpse_images = None
        for i in range(self.num_zoom_image):
            current_img = img
            glimpse_half_size = int(self.retina_size*.5 * self.scale**i)
            top = int(location[0] - glimpse_half_size)
            bottom = int(location[0] + glimpse_half_size)
            left = int(location[1] - glimpse_half_size)
            right = int(location[1] + glimpse_half_size)

            if top < 0 or left < 0 or bottom > image_size or right > image_size:
                pad_dims = (
                    glimpse_half_size, glimpse_half_size,
                    glimpse_half_size, glimpse_half_size,
                )
                current_img = F.pad(current_img, pad_dims, "constant", 0)
                top += glimpse_half_size
                bottom += glimpse_half_size
                left += glimpse_half_size
                right += glimpse_half_size
            glimpse_images = append(
                glimpse_images,
                resize(current_img[:, top:bottom, left:right],
                       self.retina_size)
            )
        return glimpse_images


class GlimpseNetwork(nn.Module):
    """
    g_t = relu(
                fc3(
                    relu(fc1(glimpse))
                ) +
                fc4(
                    relu(fc2(location))
                )
           )
    """

    def __init__(self,
                 num_hidden_glimpse,
                 num_hidden_location,
                 retina_size,
                 num_zoom_image,
                 scale,
                 channel):
        super().__init__()
        self.glimpse_sensor = GlimpseSensor(
            retina_size, num_zoom_image, scale)

        self.fc1 = nn.Linear(num_zoom_image * channel * retina_size * retina_size,
                             num_hidden_glimpse)
        self.fc2 = nn.Linear(2, num_hidden_location)
        self.fc3 = nn.Linear(num_hidden_glimpse,
                             num_hidden_glimpse+num_hidden_location)
        self.fc4 = nn.Linear(num_hidden_location,
                             num_hidden_glimpse+num_hidden_location)

    def forward(self, images, location_prev):
        glimpse = self.glimpse_sensor.foveate(images, location_prev)
        return F.relu(
            F.relu(self.fc3(
                F.relu(self.fc1(glimpse))
            )) +
            F.relu(self.fc4(
                F.relu(self.fc2(location_prev))
            ))
        )


class CoreNetwork(nn.Module):
    """
        In the paper, this function is "fh(ceta_h)"
        h_t = relu( fc1(h_t_prev) + fc2(g_t) )
    """

    def __init__(self, num_hidden_glimpse_n_location, num_hidden_core):
        super().__init__()
        self.fc1 = nn.Linear(num_hidden_glimpse_n_location, num_hidden_core)
        self.fc2 = nn.Linear(num_hidden_core, num_hidden_core)

    def forward(self, g_t, h_prev):
        return F.relu(self.fc1(g_t) + self.fc2(h_prev))


class ActionNetwork(nn.Module):
    def __init__(self, num_hidden_core, num_action):
        super().__init__()
        self.fc = nn.Linear(num_hidden_core, num_action)

    def forward(self, h_t):
        return F.log_softmax(self.fc(h_t), dim=1)


class LocationNetwork(nn.Module):
    def __init__(self, num_hidden_core, std, num_location=2):
        super().__init__()
        self.std = std
        self.fc = nn.Linear(num_hidden_core, num_location)

    def forward(self, h_t):
        # compute mean
        mu = torch.tanh(self.fc(h_t.detach()))  # TODO detach???

        # reparametrization trick
        noise = torch.zeros_like(mu)
        noise.data.normal_(std=self.std)

        # bound between [-1, 1]
        location = torch.tanh(mu + noise)

        return mu, location


class BaselineNetwork(nn.Module):
    def __init__(self, num_hidden_core, num_output):
        super().__init__()
        self.fc = nn.Linear(num_hidden_core, num_output)

    def forward(self, h_t):
        # TODO detach???
        b_t = F.relu(self.fc(h_t.detach()))
        return b_t


class RecurrentAttention(nn.Module):
    def __init__(self,
                 num_hidden_glimpse,
                 num_hidden_location,
                 num_hidden_core,
                 num_zoom_image,
                 num_glimpse,
                 num_action,
                 retina_size,
                 scale,
                 channel,
                 std):
        super().__init__()
        self.num_glimpse = num_glimpse
        self.num_hidden_core = num_hidden_core
        self.std = std

        num_hidden_glimpse_n_location = num_hidden_glimpse + num_hidden_location
        self.glimseNetwork = GlimpseNetwork(
            num_hidden_glimpse=num_hidden_glimpse,
            num_hidden_location=num_hidden_location,
            retina_size=retina_size,
            num_zoom_image=num_zoom_image,
            scale=scale,
            channel=channel)
        self.coreNetwork = CoreNetwork(
            num_hidden_glimpse_n_location=num_hidden_glimpse_n_location, num_hidden_core=num_hidden_core)
        self.locationNetwork = LocationNetwork(
            num_hidden_core=num_hidden_core, std=std)
        self.actionNetwork = ActionNetwork(
            num_hidden_core=num_hidden_core, num_action=num_action)
        self.baselineNetwork = BaselineNetwork(
            num_hidden_core=num_hidden_core, num_output=1)

    def forward(self, images):
        # TODO h_t, l_t should be random?
        # TODO rename log_pi / log_probas to make more sense

        log_pis = None
        baselines = None

        h_t = torch.zeros(images.shape[0], self.num_hidden_core, device=device)
        l_t = torch.Tensor(images.shape[0], 2, device=device).uniform_(-1, 1)

        for _ in range(self.num_glimpse):
            g_t = self.glimseNetwork(images=images, location_prev=l_t)
            h_t = self.coreNetwork(g_t=g_t, h_prev=h_t)
            mu, l_t = self.locationNetwork(h_t)
            b_t = self.baselineNetwork(h_t)
            baselines = append(baselines, b_t, dim=1)

            # we assume both dimensions are independent
            # 1. pdf of the joint is the product of the pdfs
            # 2. log of the product is the sum of the logs
            log_pi = torch.distributions.Normal(mu, self.std).log_prob(l_t)
            # This is log_prob in in Guassian Policy
            log_pi = torch.sum(log_pi, dim=1)
            log_pis = append(log_pis, log_pi)
        log_probas = self.actionNetwork(h_t)
        return log_probas, log_pis.transpose(1, 0), baselines.squeeze()

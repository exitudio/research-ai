import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from helpers import append, resize


class GlimpseSensor:
    def __init__(self, retina_size, num_glimpse=3, scale=2):
        """
        Args
        ----
        retina_size - smallest size of image
        """
        self.retina_size = retina_size
        self.num_glimpse = num_glimpse
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
        # Todo it is an inaccurate if image size is odd number.
        glimpse_images = None
        for i in range(self.num_glimpse):
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
                 num_glimpse,
                 scale,
                 channel):
        super().__init__()
        self.glimpse_sensor = GlimpseSensor(retina_size, num_glimpse, scale)

        self.fc1 = nn.Linear(num_glimpse * channel * retina_size * retina_size,
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

    def __init__(self, num_glimpse_location, num_h):
        super().__init__()
        self.fc1 = nn.Linear(num_glimpse_location, num_h)
        self.fc2 = nn.Linear(num_h, num_h)

    def forward(self, glimpse_location, h_prev):
        return F.relu(self.fc1(glimpse_location) + self.fc2(h_prev))

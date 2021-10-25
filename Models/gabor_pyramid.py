import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class GaborPyramid(nn.Module):
    """
    Create a module that maps stacks of images to a Gabor pyramid.
    Only works in grayscale
    """
    def __init__(self, 
                 nlevels=5):
        super(GaborPyramid, self).__init__()
        self.nlevels = nlevels
        self.setup()

    def setup(self):
        # The filters will be 8x1x9x9
        xi, yi = torch.meshgrid(torch.arange(-4, 5), torch.arange(-4, 5))
        filters = []
        for ii in range(4):
            coso = np.cos(ii * np.pi / 4)
            sino = np.sin(ii * np.pi / 4)
            G = torch.exp(-(xi**2+yi**2)/2/2**2)
            thefilt1 = torch.cos((coso*xi+sino*yi)*.8) * G
            thefilt2 = torch.sin((coso*xi+sino*yi)*.8) * G
            thefilt1 = thefilt1 - G / G.mean() * thefilt1.mean()
            thefilt2 = thefilt2 - G / G.mean() * thefilt2.mean()
            scale = 1 / torch.sqrt((thefilt1 ** 2).sum())

            filters += [thefilt1 * scale, thefilt2 * scale]

        downsample_filt = torch.tensor([[.25, .5, .25], [.5, 1.0, .5], [.25, .5, .25]]).view(1, 1, 3, 3)
        downsample_filt /= 4.0

        filters = torch.stack(filters, dim=0).view(8, 1, 9, 9)
        self.register_buffer('filters', filters, False)
        self.register_buffer('downsample_filt', downsample_filt, False)

    def forward(self, X):
        X_ = X.sum(axis=1, keepdims=True)
        maps = []
        for i in range(self.nlevels):
            outputs = F.conv2d(X_, self.filters, padding=4)
            magnitude = torch.sqrt((outputs ** 2)[:, ::2, :, :] + 
                                   (outputs ** 2)[:, 1::2, :, :])
            if i == 0:
                maps.append(magnitude)
            else:
                maps.append(F.interpolate(magnitude, scale_factor=2**i, mode='bilinear', align_corners=False)[:, :, :X.shape[2], :X.shape[3]])

            X_ = F.conv2d(X_, self.downsample_filt, padding=1, stride=2)
        
        return torch.cat(maps, axis=1)
            

class GaborPyramid3d(nn.Module):
    """
    Create a module that maps stacks of images to a 3d Gabor pyramid.
    Only works in grayscale
    """
    def __init__(self, 
                 nlevels=5,
                 nt=5, #7
                 stride=1,
                 motionless=False):
        super(GaborPyramid3d, self).__init__()
        self.nt = nt
        self.nlevels = nlevels
        self.stride = stride
        self.motionless = motionless
        self.setup()

    def setup(self):
        # The filters will be 8x1xntx9x9
        nx, no = 9, 4
        zi, yi, xi = torch.meshgrid(torch.arange(-(self.nt // 2), 
                                                 (self.nt + 1) // 2), 
                                    torch.arange(-4, 5), 
                                    torch.arange(-4, 5))
        zi, yi, xi = zi.to(dtype=torch.float), yi.to(dtype=torch.float), xi.to(dtype=torch.float)

        assert zi.shape[0] == self.nt
        filters = []
        for ii in range(no):
            for dt in [-1, 0, 1]:
                coso = np.cos(ii * np.pi / no)
                sino = np.sin(ii * np.pi / no)
                G = torch.exp(-(xi**2 + yi**2 + (zi / self.nt * nx)**2)/2/2**2)
                thefilt1 = torch.cos((coso * xi + sino * yi)*.8 + dt * zi / self.nt * np.pi * 2) * G
                thefilt2 = torch.sin((coso * xi + sino * yi)*.8 + dt * zi / self.nt * np.pi * 2) * G
                thefilt1 = thefilt1 - G / G.mean() * thefilt1.mean()
                thefilt2 = thefilt2 - G / G.mean() * thefilt2.mean()
                scale = 1 / torch.sqrt((thefilt1 ** 2).sum())

                filters += [thefilt1 * scale, thefilt2 * scale]

        downsample_filt = torch.tensor([[.25, .5, .25], [.5, 1.0, .5], [.25, .5, .25]]).view(1, 1, 3, 3)
        downsample_filt /= 4.0

        filters = torch.stack(filters, dim=0).view(no * 3 * 2, 1, self.nt, nx, nx)
        self.register_buffer('filters', filters, False)
        self.register_buffer('downsample_filt', downsample_filt, False)

    def forward(self, X):
        # Transform to grayscale.
        X_ = X.sum(axis=1, keepdims=True)
        maps = []
        for i in range(self.nlevels):
            outputs = F.conv3d(X_, 
                               self.filters, 
                               padding=(self.nt//2, 4, 4),
                               stride=self.stride)
            magnitude = torch.sqrt((outputs ** 2)[:, ::2, :, :, :] + 
                                   (outputs ** 2)[:, 1::2, :, :, :])

            if self.motionless:
                # Add the two directions together
                magnitude = torch.cat([(magnitude[:, 0::3, :, :, :] + magnitude[:, 2::3, :, :, :]) / 2.0,
                                        magnitude[:, 1::3, :, :, :]], axis=1)

            if i == 0:
                maps.append(magnitude)
            else:
                # Only the spatial dimension is resized.
                the_map = F.interpolate(magnitude.reshape((magnitude.shape[0], -1, magnitude.shape[-2], magnitude.shape[-1])), 
                                        scale_factor=2**i, 
                                        mode='bilinear', 
                                        align_corners=False)
                the_map = the_map.reshape(magnitude.shape[0], 
                                          magnitude.shape[1], -1, the_map.shape[-2], the_map.shape[-1])[:, :, :, :X.shape[-2], :X.shape[-1]]
                maps.append(the_map)
        
            X_ = F.conv2d(X_.reshape((X_.shape[0]*X_.shape[2], 1, X_.shape[-2], X_.shape[-1])), 
                    self.downsample_filt, 
                    padding=1, 
                    stride=2)
            X_ = X_.reshape(X.shape[0], 1, -1, X_.shape[-2], X_.shape[-1])

        return torch.cat(maps, axis=1) # [:, :, 2:-2, :, :]
    
if __name__ == "__main__":
    
    import time
    
    mydata = torch.FloatTensor(8, 3, 5, 64, 64).to('cpu')
    nn.init.normal_(mydata)
    
    tic = time.time()
    gabor_pyramid = GaborPyramid3d().to('cpu')
    
    agg_out = gabor_pyramid(mydata)
    print(agg_out.shape)

    print(time.time()-tic)
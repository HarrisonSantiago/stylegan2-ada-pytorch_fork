import os.path

import torchvision
import numpy as np
import math
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import lpips
torch.set_printoptions(precision=5)
from torch import nn
import matplotlib.pyplot as plt
from torch_utils import misc

from utils import *
import copy

#EXAMPLES
#w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
#img = G.synthesis(w, noise_mode='const', force_fp32=True)

#def make_noise(device : torch.device, bs=1):
#   noises = [torch.randn(bs, 1, 2 ** 2, 2 ** 2, device=device)]

#   for i in range(3, self.log_size + 1):
#       for _ in range(2):
#           noises.append(torch.randn(bs, 1, 2 ** i, 2 ** i, device=device))

#   return noises

def get_transformation(image_size):
    return transforms.Compose(
        [transforms.Resize(image_size),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def loss_geocross(latent):
    if latent.size() == (1, 512):
        return 0
    else:
        num_latents = latent.size()[1]
        X = latent.view(-1, 1, num_latents, 512)
        Y = latent.view(-1, num_latents, 1, 512)
        A = ((X - Y).pow(2).sum(-1) + 1e-9).sqrt()
        B = ((X + Y).pow(2).sum(-1) + 1e-9).sqrt()
        D = 2 * torch.atan2(A, B)
        D = ((D.pow(2) * 512).mean((1, 2)) / 8.).mean()
        return D

class SphericalOptimizer():
    def __init__(self, params):
        self.params = params
        with torch.no_grad():
            self.radii = {param: (param.pow(2).sum(tuple(range(2,param.ndim)), keepdim=True)+1e-9).sqrt() for param in params}
    @torch.no_grad()
    def step(self, closure=None):
        for param in self.params:
            param.data.div_((param.pow(2).sum(tuple(range(2,param.ndim)), keepdim=True)+1e-9).sqrt())
            param.mul_(self.radii[param])

class LatentOptimizer(torch.nn.Module):
    def __init__(self, config, Generator, device=torch.device):
        super().__init__()
        self.config = config

        self.G = copy.deepcopy(Generator).eval().requires_grad_(False).to(device)


        if config['image_size'][0] != config['image_size'][1]:
            raise Exception('Non-square images are not supported yet.')


        self.G.start_layer = config['start_layer']
        self.G.end_layer = config['end_layer']

        #TODO: replace this mapping proxy with the mapping in ada


        self.init_state()

    def init_state(self):
        device = self.config['device']
	  #Project latent to unit ball?
        self.project = self.config["project"]
	  #How many steps to run per layer
        self.steps = self.config["steps"]

        self.layer_in = None
        self.best = None
        self.current_step = 0


        # save filters
	  #Changed by removing perc, m, self.indices, self.filters, self.sign_pattern stuff
	  #Looks like it only applies to the partial circulant compressed sensing stuff



    def get_lr(self, t, initial_lr, rampdown=0.75, rampup=0.05):
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)
        return initial_lr * lr_ramp

    def step4(self, block_ws, z_p, gen_img, target_exc, start_res, radius):

        lr = 0.05
        z_p = z_p.clone().detach()
        new = torch.tensor(z_p,dtype=torch.float32, device="cuda", requires_grad=True).cuda()

        optimizer4 = optim.Adam([new], lr=lr)
        loss_fcn = nn.MSELoss()

        steps = 50
        loss_tracker = []
        mse_min = np.inf
        for _ in range(steps):
            holder = new.clone()

            #deviation = project_onto_l1_ball(new - z_p, radius)
            #new = (z_p + deviation)

            z, img = self.run_G2(block_ws, new, gen_img, start_res)


            img = (img * 127.5 + 128).clamp(0, 255)

            # int_cone_exc = ISETBIO(gen_img)
            int_cone_exc = img

            loss = loss_fcn(int_cone_exc[0], target_exc)

            loss_tracker.append(loss.detach().cpu())
            optimizer4.zero_grad()
            loss.backward(retain_graph=True)
            optimizer4.step()

            if loss < mse_min:
                mse_min = loss
                best_z = new
                best_img = gen_img

        #4 project to l1 ball

        y = np.array(loss_tracker)
        x = np.arange(len(y))

        plt.plot(x, y)
        plt.xlabel('steps')
        plt.ylabel('G_2(x) - x\' loss')
        plt.title('Step 4, res: '+ str(start_res))
        plt.show()

        #print('comparing')
        #print(best_z[:15])
        #print(z_p[:15])

        return best_z, best_img, mse_min

    def step5(self, z_p_sq, z_k,  current_res, z_p_sq_ws,  initial_learning_rate = 0.005):
        print('--- starting step 5 ---')
        #z_k = z_k.clone().detach()
        #holder = torch.tensor(z_k, dtype=torch.float32, device="cuda", requires_grad=True).cuda()

        num_steps = 600

        holder = torch.randn([1, self.G.z_dim], dtype=torch.float32, device="cuda", requires_grad=True).cuda()

        optimizer5 = torch.optim.Adam([holder], lr=initial_learning_rate)

        loss_min = np.inf
        loss_tracker = []

        for step in range(num_steps):
            #print('cur res: ', current_res)
            _, z, gen_img, ws_gen = self.run_G1(holder, current_res)

            #print('z shape: ', z.shape)
            #print('z_k_hat shape: ', z_p_sq.shape )
            loss = torch.sum(torch.square(z_p_sq_ws - ws_gen))

            #print('loss: ', loss)
            optimizer5.zero_grad()
            loss.backward()
            optimizer5.step()
            loss_tracker.append(torch.sum(torch.square(z_p_sq - z)).detach().cpu())

            if (loss < loss_min):
                loss_min = loss
                z_k_hat = holder
                img = gen_img

        y = np.array(loss_tracker)
        x = np.arange(len(y))

        plt.plot(x, y)
        plt.xlabel('steps')
        plt.ylabel('G_1(z^k) - z^p\' loss')
        plt.title('Step 5, res: ' + str(current_res))
        plt.show()
        print('step 5 loss: ', loss_min)
        return z_k_hat, img


    def run_G1(self, z_k,  end_res):

        #holder = torch.ones(z_k.shape, device = "cuda", requires_grad = True)
        #holder = holder * z_k.clone()
        #ws = self.G.mapping(holder, None)
        ws = self.G.mapping(z_k, None)

        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.G.num_ws, self.G.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.G.synthesis.block_resolutions:  # up to certain layer
                block = getattr(self.G.synthesis, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv


        z = img = None
        for res, cur_ws in zip(self.G.synthesis.block_resolutions, block_ws):
            block = getattr(self.G.synthesis, f'b{res}')

            z, img = block(z, img, cur_ws, noise_mode = 'const')

            if res == end_res:
                break

        return block_ws, z, img, ws #this is some z_p

    def run_G2(self, block_ws, z_k, gen_img, start_res):

        start = False

        for res, cur_ws in zip(self.G.synthesis.block_resolutions, block_ws):
            if start:
                block = getattr(self.G.synthesis, f'b{res}')
                z_k, gen_img = block(z_k, gen_img, cur_ws, {})


            if res == start_res:
                start = True


        return z_k, gen_img #completed image


    def invert_(self, z_k_hat, z_k_hat_img, target_exc, current_res, radius = 250):
        torch.autograd.set_detect_anomaly(True)
        #step 2
        block_ws, z_p_hat, z_p_hat_img, ws = self.run_G1(z_k_hat, current_res)


        #steps 3-6
        radius = 100
        pbar = tqdm(range(radius))
        mse_max = np.inf
        for i in pbar:

            #step 4
            #Does w need to be redone for the z_p
            z_p_sq , z_p_sq_im, loss = self.step4(block_ws, z_p_hat, z_p_hat_img, target_exc, current_res, i+1)
            if i > 0:
                z_p_sq , z_p_sq_im, loss = self.step4(block_ws, z_p_hat_new, z_p_hat_img, target_exc, current_res, i+1)
            print('step 4 loss: ', loss)

            #step 5
            z_k_hat_new, img = self.step5(z_p_sq, z_k_hat, current_res, ws.detach().clone())

            if loss < mse_max:
                mse_max = loss
                best_zk = z_k_hat_new
                best_img = z_p_sq_im


            #step 6

            block_ws, z_p_hat_new, img, ws = self.run_G1(z_k_hat_new, current_res)

            z_p_hat_img = img

            z, img = self.run_G2(block_ws, z_p_hat_new, img, current_res)


            im = self.genToPng(img)
            name = str(current_res) + "/" + str(i) + ".png"
            im.save(name)

        return block_ws, best_zk, best_img




    def step1(self, target_exc, num_steps = 100, initial_learning_rate = 0.1):
        print('--- step 1 ---')
        loss_tracker = []

        z_k = torch.randn([1, self.G.z_dim], dtype = torch.float32, device = "cuda", requires_grad=True).cuda()

        #print('step 1 z_k_hat shape: ', z_k.shape)

        optimizer = torch.optim.Adam([z_k], lr = initial_learning_rate)
        loss_fcn = nn.MSELoss()

        mse_min = np.inf

        for step in range(num_steps):
            gen_img = self.G(z_k, c=None, noise_mode='const')
            gen_img = (gen_img * 127.5 + 128).clamp(0, 255)

            #gen_exc = ISETBio[]
            gen_exc = gen_img

            #print('shape: ', gen_exc.shape)
            loss = loss_fcn(gen_exc[0], target_exc)

            #loss = (target_exc - gen_exc[0]).square().sum()
            #print('step: ', step, ', loss: ', loss)
            loss_tracker.append(loss.detach().cpu())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if loss < mse_min:
                mse_min = loss
                z_k_hat = z_k

        y = np.array(loss_tracker)
        x = np.arange(len(y))

        plt.plot(x, y)
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.show()
        return z_k_hat, gen_img

    def genToPng(self, gen_img):
        #turns it from something that G(z,c) outputs, into a png savable format
        img = (gen_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        im = Image.fromarray(img[0].cpu().numpy(), 'RGB')

        return im

    def reconstruct(self, target_image):
        #print('Running with the following config....')
        #pretty(self.config)

        #target_exc = []
        target_exc = target_image

        #step 1
        z_k_hat, img = self.step1(target_exc)

        #Can remove later
        print('Saving image')
        img = self.G(z_k_hat, None)
        im = self.genToPng(img)
        im.save('step1.png')

        #This is how we get the layers to go over
        res_lst = self.G.synthesis.block_resolutions

        for i, res in enumerate(res_lst):
            if not os.path.isdir(str(res)):
                os.mkdir(str(res))
            os.remove(res+'/*')
            print('starting layer ', i, 'with a resolution of ', str(res))
            block_ws, z_k_hat, gen_img = self.invert_(z_k_hat, img, target_exc, res)




        print('outputting best image')
        img = self.G(z_k_hat, None)
        im = self.genToPng(img)
        im.save('final.png')
        return z_k_hat, target_image

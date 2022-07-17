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


    def get_lr(self, t, initial_lr, rampdown=0.75, rampup=0.05):
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)
        return initial_lr * lr_ramp

    def step4(self, block_ws, int_latent_p, gen_img, target_exc, start_res, radius):

        lr = 0.05
        int_latent_p = int_latent_p.clone().detach()
        new = torch.tensor(int_latent_p,dtype=torch.float32, device="cuda", requires_grad=True).cuda()

        optimizer4 = optim.Adam([new], lr=lr)
        loss_fcn = nn.MSELoss()

        steps = 150
        loss_tracker = []
        diff_tracker = []
        mse_min = np.inf
        for _ in range(steps):

            #deviation = project_onto_l1_ball(new - z_p, radius)
            #new = (z_p + deviation)

            last_latent, img = self.run_G2(block_ws, new, gen_img, start_res)


            img = (img * 127.5 + 128).clamp(0, 255)

            # int_cone_exc = ISETBIO(gen_img)
            int_cone_exc = img

            loss = loss_fcn(int_cone_exc[0], target_exc) + 0.1 * torch.sum(torch.square(new - int_latent_p))

            loss_tracker.append(loss.detach().cpu())
            optimizer4.zero_grad()
            loss.backward(retain_graph=True)
            optimizer4.step()
            if loss < mse_min:
                mse_min = loss
                best_int_latent_p = new
                best_img = gen_img
            diff_tracker.append(torch.sum(torch.square(best_int_latent_p - int_latent_p)).detach().cpu())


        #4 project to l1 ball

        y = np.array(loss_tracker)
        x = np.arange(len(y))

        plt.plot(x, y)
        plt.xlabel('steps')
        plt.ylabel('G_2(x) - x\' loss')
        plt.title('Step 4, res: '+ str(start_res))
        plt.show()

        y = np.array(diff_tracker)
        x = np.arange(len(y))

        plt.plot(x, y)
        plt.xlabel('steps')
        plt.ylabel('new latent - old latent')
        plt.title('Step 4, latent differences: ' + str(start_res))
        plt.show()


        print('total diff between old latent and new latent: ', torch.sum(best_int_latent_p - int_latent_p))

        return best_int_latent_p, best_img, mse_min

    def step5(self, target_int_latent, step2_latent_k, current_res, initial_learning_rate = 0.005):
        print('--- starting step 5 ---')
        latent_k = step2_latent_k.clone().detach()
        holder = torch.tensor(latent_k, dtype=torch.float32, device="cuda", requires_grad=True).cuda()

        num_steps = 800

        #holder = torch.randn([1, self.G.z_dim], dtype=torch.float32, device="cuda", requires_grad=True).cuda()

        optimizer5 = torch.optim.Adam([holder], lr=initial_learning_rate)

        loss_min = np.inf
        loss_tracker = []

        for step in range(num_steps):
            #print('cur res: ', current_res)
            block_ws, int_latent, img = self.run_G1(holder, current_res)

            #print('z shape: ', z.shape)
            #print('z_k_hat shape: ', z_p_sq.shape )
            loss = torch.sum(torch.square(int_latent - target_int_latent))

            #print('loss: ', loss)
            optimizer5.zero_grad()
            loss.backward()
            optimizer5.step()
            loss_tracker.append(loss.detach().cpu())

            if (loss < loss_min):
                loss_min = loss
                latent_k_hat = holder
                gen_img = img

        y = np.array(loss_tracker)
        x = np.arange(len(y))

        plt.plot(x, y)
        plt.xlabel('steps')
        plt.ylabel('G_1(z^k) - z^p\' loss')
        plt.title('Step 5, res: ' + str(current_res))
        plt.show()
        print('step 5 loss: ', loss_min)
        return latent_k_hat, gen_img


    def run_G1(self, w_k,  end_res):

        #holder = torch.ones(z_k.shape, device = "cuda", requires_grad = True)
        #holder = holder * z_k.clone()
        #ws = self.G.mapping(holder, None)

        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(w_k, [None, self.G.num_ws, self.G.w_dim])
            ws = w_k.to(torch.float32)
            w_idx = 0
            for res in self.G.synthesis.block_resolutions:  # up to certain layer
                block = getattr(self.G.synthesis, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv


        int_latent = img = None
        for res, cur_ws in zip(self.G.synthesis.block_resolutions, block_ws):
            block = getattr(self.G.synthesis, f'b{res}')

            int_latent, img = block(int_latent, img, cur_ws, {})

            if res == end_res:
                break

        return block_ws, int_latent, img #this is some z_p

    def run_G2(self, block_ws, int_latent_p, gen_img, start_res):

        start = False

        for res, cur_ws in zip(self.G.synthesis.block_resolutions, block_ws):
            if start:
                block = getattr(self.G.synthesis, f'b{res}')
                int_latent_p, gen_img = block(int_latent_p, gen_img, cur_ws, {})


            if res == start_res:
                start = True

        #final int_latent_p is the for the last layer
        return int_latent_p, gen_img #completed image


    def invert_(self, w_k_hat, w_k_hat_img, target_exc, current_res, radius = 250):
        torch.autograd.set_detect_anomaly(True)
        #step 2
        block_ws, int_latent_p_hat, step2_img = self.run_G1(w_k_hat, current_res)

        best_latentk = w_k_hat
        #steps 3-6
        radius = 10
        pbar = tqdm(range(radius))
        mse_max = np.inf
        for i in pbar:

            #step 4
            #Does w need to be redone for the z_p

            if i > 0:
                best_int_latent_p, best_int_lat_p_img, mse_min = self.step4(block_ws, step6_p_hat, step6_img,
                                                                            target_exc, current_res, i+1)
            else:
                best_int_latent_p, best_int_lat_p_img, loss = self.step4(block_ws, int_latent_p_hat, step2_img,
                                                                         target_exc, current_res, i + 1)

            print('step 4 loss: ', loss)

            #step 5
            latent_k_hat_new, step5_img = self.step5(best_int_latent_p, w_k_hat, current_res)




            if loss < mse_max:
                mse_max = loss
                best_latentk = latent_k_hat_new
                best_img = step5_img


            #step 6

            block_ws_s6, step6_p_hat, step6_img = self.run_G1(best_latentk, current_res)



            _, round_img = self.run_G2(block_ws_s6, step6_p_hat, step6_img, current_res)


            im = self.genToPng(round_img)
            name = str(current_res) + "/" + str(i) + ".png"
            im.save(name)

        return block_ws, best_latentk, best_img




    def step1(self, target_exc, num_steps = 100, initial_learning_rate = 0.1, w_avg_samples = 10000 ):
        print('--- step 1 ---')
        loss_tracker = []

        z_samples = np.random.RandomState(123).randn(w_avg_samples, self.G.z_dim)
        w_samples = self.G.mapping(torch.from_numpy(z_samples).to("cuda"), None)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)




        #print('step 1 z_k_hat shape: ', z_k.shape)

        w_opt = torch.tensor(w_avg, dtype=torch.float32, device="cuda", requires_grad=True)
        optimizer = torch.optim.Adam([w_opt] , betas=(0.9, 0.999), lr=initial_learning_rate)
        loss_fcn = nn.MSELoss()

        mse_min = np.inf

        for step in range(num_steps):
            ws = w_opt.repeat([1, self.G.mapping.num_ws, 1])
            gen_img = self.G.synthesis(ws, noise_mode='const')
            gen_img = (gen_img * 127.5 + 128).clamp(0, 255)

            #gen_exc = ISETBio[]
            gen_exc = gen_img

            #print('shape: ', gen_exc.shape)
            #print('t shape: ', target_exc.shape)
            loss = loss_fcn(gen_exc[0], target_exc)


            loss_tracker.append(loss.detach().cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if loss < mse_min:
                mse_min = loss
                best_w = ws
                best_img = gen_img

        y = np.array(loss_tracker)
        x = np.arange(len(y))

        plt.plot(x, y)
        plt.xlabel('steps')
        plt.ylabel('loss')
        plt.show()


        return best_w, best_img

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
        w_k_hat, img = self.step1(target_exc)

        #Can remove later
        print('Saving image')
        img = self.G.synthesis(w_k_hat, None)
        im = self.genToPng(img)
        im.save('step1.png')

        #This is how we get the layers to go over
        res_lst = self.G.synthesis.block_resolutions

        for i, res in enumerate(res_lst):
            if not os.path.isdir(str(res)):
                os.mkdir(str(res))
            os.remove(res+'/*')
            print('starting layer ', i, 'with a resolution of ', str(res))
            w_k_hat, gen_img = self.invert_(w_k_hat, img, target_exc, res)




        print('outputting best image')
        img = self.G(w_k_hat, None)
        im = self.genToPng(img)
        im.save('final.png')
        return w_k_hat, target_image

    def alternate_solver(self, ws, target_exc):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.G.num_ws, self.G.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.G.synthesis.block_resolutions:  # up to certain layer
                block = getattr(self.G.synthesis, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv


        block_res = self.G.synthesis.block_resolutions
        for res, i in zip(block_res, range(len(block_ws))): # this is the res we optimize over

            #generate up to the current res using optimal ws
            x = img = None
            for res1, cur_w1 in zip(block_res, block_ws):
                if res1 < res:
                    print(res1)
                    print(cur_w1.shape)
                    block = getattr(self.G.synthesis, f'b{res}')
                    x, img = block(x, img, cur_w1, {})
                else:
                    break

            targ_w = block_ws[i]
            holder = block_ws[i].clone().detach()
            holder = torch.tensor(holder, device = "cuda", requires_grad = True)


            optim = torch.optim.Adam([holder], lr = 0.05)
            max_loss = np.inf
            loss_tracker = []
            for _ in range(100):
                gen_img = self.inner(holder, block_res, block_ws, res, x, img)
                gen_exc = gen_img
                
                loss = torch.sum(torch.square(target_exc - gen_exc))
                loss_tracker.append(loss.detach().cpu())

                if loss < max_loss:
                    best_w = targ_w
                    max_loss = loss
                    best_img = gen_img

                optim.zero_grad()
                loss.backward()
                optim.step()

            block_ws[i] = best_w

        return best_img



    def inner(self, targ_w, block_res, block_ws, res_start, x, img):
        #modified G2, returns the block w value and gen img


        for res, cur_ws in zip(block_res, block_ws):
            if res == res_start:
                block = getattr(self.G.synthesis, f'b{res}')
                x, img = block(x, img, targ_w, {})
            if res > res_start:
                block = getattr(self.G.synthesis, f'b{res}')
                x, img = block(x, img, cur_ws, {})


        return img


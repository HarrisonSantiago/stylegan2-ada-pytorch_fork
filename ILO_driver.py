import os.path
import pytorch_ssim
from mp4_gen import *
import torchvision
from torchvision.utils import save_image
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
import matlab.engine
import lpips
import scipy.io as sio
import imageio



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
    def __init__(self, config, Generator, im_width, device=torch.device):
        super().__init__()
        self.config = config
        #---creates matlab engine---
        self.engine = matlab.engine.start_matlab()

        self.home_dir = self.engine.pwd()
        self.engine.init(self.home_dir, im_width, nargout = 0) #loads ISETBio stuff and creates the retina object
        self.engine.cd(self.home_dir)
        self.retinaPath = self.home_dir+ "/retina"+im_width+".mat"
        self.coneInvPath = self.home_dir+ "/render_pinv"+im_width+".mat"
        self.renderPath = self.home_dir+ "/render"+im_width+".mat"
        self.coneInv = torch.tensor(sio.loadmat(self.coneInvPath)['render_pinv'], dtype = torch.float32, device = "cuda")
        self.render = torch.tensor(sio.loadmat(self.renderPath)['render'], dtype = torch.float32, device = "cuda")
        self.visualPath = "for_mp4.png"
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




    def step1(self, targ_path, num_steps = 100, initial_learning_rate = 0.1, w_avg_samples = 10000 ):
        print('--- step 1 ---')

        self.targ_exc = torch.tensor(np.asarray(self.engine.getConeResp(targ_path, self.retinaPath)),
                                     dtype=torch.float32, device="cuda")

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

            img = self.G.synthesis(ws, noise_mode='const', force_fp32=True)
            im = self.genToPng(img)
            im.save('current_guess.png')

            gen_exc = torch.tensor(np.asarray(self.engine.getConeResp('current_guess.png', self.retinaPath)),
                                   dtype=torch.float32, device = "cuda", requires_grad = True)

            #print('shape: ', gen_exc.shape)
            #print('t shape: ', target_exc.shape)
            loss = loss_fcn(gen_exc, self.targ_exc)


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

        print('num ws: ', self.G.mapping.num_ws)
        print('best_w shape', best_w.shape)

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

    def alternate_solver(self, ws):

        loss_fcn = nn.MSELoss()
        mse_min = np.inf
        loss_tracker = []
        num_steps = 300
        ws = ws.detach().clone()

        for i in range(1,ws.shape[1]-1):

            w_opt = torch.tensor(ws[0,i], dtype=torch.float32, device="cuda", requires_grad=True)

            optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=0.05)

            beg = torch.unsqueeze(ws[0, :i], dim = 0)
            end = torch.unsqueeze(ws[0, i + 1:], dim =0)

            for step in range(num_steps):
                mid = torch.unsqueeze(torch.unsqueeze(w_opt, dim=0), dim=0)

                to_synt = torch.cat((beg, mid, end), dim = 1)

                gen_img = self.G.synthesis(to_synt, noise_mode='const')
                im = self.genToPng(gen_img)
                im.save('curr_guess.png')


                #current
                gen_exc = torch.tensor(np.asarray(self.engine.getConeResp('curr_guess.png', self.retinaPath)),
                                       dtype=torch.float32, device="cuda", requires_grad=True)
                loss = loss_fcn(gen_exc, self.targ_exc)

                #---- Start of what worked for sure, not actually using exc
                #gen_img = (gen_img * 127.5 + 128).clamp(0, 255)
                #gen_exc = gen_img
#
                #loss = loss_fcn(gen_exc[0], target_exc)
                #----- End of what worked for sure

                loss_tracker.append(loss.detach().cpu())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('loss', loss)

                if loss < mse_min:
                    mse_min = loss
                    best_w = to_synt[0,i].detach().clone()
                    best_img = gen_img

            plt.plot(loss_tracker)
            plt.show()

            ws[0,i] = best_w

            img = self.G.synthesis(ws, noise_mode='const', force_fp32=True)
            im = self.genToPng(img)
            im.save(str(i) +'.png')

        return best_img



    def project(self, targ_img):
        self.targ_img = torch.tensor(targ_img, dtype = torch.float32, device = "cuda")

        step1_ws = self.project_step1()

        best_w, best_img = self.layer_solver(step1_ws)

        return best_w, best_img


    def project_step1(self, w_avg_samples = 10000, initial_learning_rate = .05):
        z_samples = np.random.RandomState(123).randn(w_avg_samples, self.G.z_dim)
        w_samples = self.G.mapping(torch.from_numpy(z_samples).to("cuda"), None)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)

        w_opt = torch.tensor(w_avg, dtype=torch.float32, device="cuda", requires_grad=True)
        optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)
        loss_fcn = nn.MSELoss()
        #loss_fcn1 = lpips.LPIPS(net ='alex')
        #loss_fcn1.cuda()
        #ssim_loss = pytorch_ssim.SSIM()
        mse_min = np.inf

        loss_tracker = []

        for step in range(200):
            ws = w_opt.repeat([1, self.G.mapping.num_ws, 1])

            img = self.G.synthesis(ws, noise_mode='const', force_fp32=True)
            gen_img = (img * 127.5 + 128).clamp(0, 255)


            #for MSELoss
            loss = 0.5 * loss_fcn(gen_img[0], self.targ_img)
            #loss += torch.squeeze(loss_fcn1.forward(gen_img[0], self.targ_img))
            #loss = - ssim_loss(gen_img, torch.unsqueeze(self.targ_img, dim = 0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_tracker.append(loss.detach().cpu())
            if loss < mse_min:
                mse_min = loss
                best_w = ws

        img = self.G.synthesis(best_w, noise_mode='const', force_fp32=True)
        im = self.genToPng(img)
        im.save('step1.png')

        plt.plot(loss_tracker)
        plt.show()

        return best_w

    def layer_solver(self, ws):

        loss_fcn = nn.MSELoss()
        loss_fcn1 = lpips.LPIPS(net='alex')
        loss_fcn1.cuda()
        ssim_loss = pytorch_ssim.SSIM()
        mse_min = np.inf
        num_steps = 350
        ws = ws.detach().clone()

        for i in range(0, ws.shape[1]):
            loss_tracker = []

            w_opt = torch.tensor(ws[0, i], dtype=torch.float32, device="cuda", requires_grad=True)

            optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=0.05)

            beg = torch.unsqueeze(ws[0, :i], dim=0)
            end = torch.unsqueeze(ws[0, i + 1:], dim=0)

            for step in range(num_steps):
                mid = torch.unsqueeze(torch.unsqueeze(w_opt, dim=0), dim=0)
                to_synt = torch.cat((beg, mid, end), dim=1)

                gen_img = self.G.synthesis(to_synt, noise_mode='const')
                gen_img = (gen_img * 127.5 + 128).clamp(0, 255)

                # for MSELoss
                loss = 0.6 * loss_fcn(gen_img[0], self.targ_img)
                loss += 1.2 * torch.squeeze(loss_fcn1.forward(gen_img[0], self.targ_img))
                loss += 80 * - ssim_loss(gen_img, torch.unsqueeze(self.targ_img, dim = 0))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss < mse_min:
                    mse_min = loss
                    best_w = to_synt[0, i].detach().clone()
                    best_img = gen_img
                loss_tracker.append(loss.detach().cpu())

            img = self.G.synthesis(ws, noise_mode='const', force_fp32=True)
            im = self.genToPng(img)
            im.save(str(i) + '.png')

            plt.plot(loss_tracker)
            plt.show()
            ws[0, i] = best_w

        img = self.G.synthesis(ws, noise_mode='const', force_fp32=True)
        im = self.genToPng(img)
        im.save('best_proj.png')

        return ws, best_img


    def recon_useInv(self, targ_path):

        #---start---
        linear_image = torch.tensor(np.asarray(self.engine.getImageLinear(targ_path, self.retinaPath)),
                                     dtype=torch.float32, device="cuda")
        flat = torch.flatten(linear_image)
        coneExc = torch.matmul(self.render, flat)
        targ_rec = torch.matmul(self.coneInv, coneExc)
        targ_rec = torch.reshape(targ_rec, (32, 32, 3))
        targ_rec = targ_rec.permute((2,0,1))
        #---reconstruction---

        save_image(targ_rec, 'target.png')

        best_w = self.useInv_step1(targ_rec)

        a, b = self.layer_useInv(best_w, targ_rec)

        return 0

    def useInv_step1(self, targ_img, w_avg_samples = 10000, initial_learning_rate = 0.05):

        z_samples = np.random.RandomState(123).randn(w_avg_samples, self.G.z_dim)
        w_samples = self.G.mapping(torch.from_numpy(z_samples).to("cuda"), None)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)

        w_opt = torch.tensor(w_avg, dtype=torch.float32, device="cuda", requires_grad=True)
        optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)
        loss_fcn = nn.MSELoss()
        # loss_fcn1 = lpips.LPIPS(net ='alex')
        # loss_fcn1.cuda()
        # ssim_loss = pytorch_ssim.SSIM()
        mse_min = np.inf

        loss_tracker = []

        for step in range(50):
            ws = w_opt.repeat([1, self.G.mapping.num_ws, 1])

            img = self.G.synthesis(ws, noise_mode='const', force_fp32=True)

            gen_png = self.genToPng(img)
            path = 'current_guess.png'
            gen_png.save(path)
            # ---start---
            linear_image = torch.tensor(np.asarray(self.engine.getImageLinear(path, self.retinaPath)),
                                        dtype=torch.float32, device="cuda")
            flat = torch.flatten(linear_image)
            coneExc = torch.matmul(self.render, flat)
            gen_rec = torch.matmul(self.coneInv, coneExc)
            gen_rec = torch.reshape(gen_rec, (32, 32, 3))
            gen_rec = gen_rec.permute((2, 0, 1))
            # ---reconstruction---

            img = torch.squeeze(img)
            #want to adjust img to gen_rec without loosing the comp map
            # img = gen_rec + c
            #gen_rec + c = img
            # c = img - gen_rec
            # img - c = gen_re

            c = img.detach().clone() - gen_rec.detach().clone()
            img -= c



            # for MSELoss
            loss = loss_fcn(img, targ_img)
            # loss += torch.squeeze(loss_fcn1.forward(gen_img[0], self.targ_img))
            # loss = - ssim_loss(gen_img, torch.unsqueeze(self.targ_img, dim = 0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_tracker.append(loss.detach().cpu())
            if loss < mse_min:
                mse_min = loss
                best_w = ws

        img = self.G.synthesis(best_w, noise_mode='const', force_fp32=True)
        im = self.genToPng(img)
        im.save('step1_inv.png')

        plt.plot(loss_tracker)
        plt.show()

        return best_w

    def layer_useInv(self, ws, targ_img):
        loss_fcn = nn.MSELoss()
        #loss_fcn1 = lpips.LPIPS(net='alex')
        #loss_fcn1.cuda()
        #ssim_loss = pytorch_ssim.SSIM()
        mse_min = np.inf
        num_steps = 50
        ws = ws.detach().clone()

        for i in range(0, ws.shape[1]):
            loss_tracker = []

            w_opt = torch.tensor(ws[0, i], dtype=torch.float32, device="cuda", requires_grad=True)

            optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=0.05)

            beg = torch.unsqueeze(ws[0, :i], dim=0)
            end = torch.unsqueeze(ws[0, i + 1:], dim=0)

            for step in range(num_steps):
                mid = torch.unsqueeze(torch.unsqueeze(w_opt, dim=0), dim=0)
                to_synt = torch.cat((beg, mid, end), dim=1)

                img = self.G.synthesis(to_synt, noise_mode='const')
                gen_png = self.genToPng(img)
                path = 'current_guess.png'
                gen_png.save(path)
                # ---start---
                linear_image = torch.tensor(np.asarray(self.engine.getImageLinear(path, self.retinaPath)),
                                            dtype=torch.float32, device="cuda")
                flat = torch.flatten(linear_image)
                coneExc = torch.matmul(self.render, flat)
                gen_rec = torch.matmul(self.coneInv, coneExc)
                gen_rec = torch.reshape(gen_rec, (32, 32, 3))
                gen_rec = gen_rec.permute((2, 0, 1))
                # ---reconstruction---

                img = torch.squeeze(img)
                # want to adjust img to gen_rec without loosing the comp map
                # img = gen_rec + c
                # gen_rec + c = img
                # c = img - gen_rec
                # img - c = gen_re

                c = img.detach().clone() - gen_rec.detach().clone()
                img -= c

                # for MSELoss
                loss = loss_fcn(img, targ_img)
                #loss += 1.2 * torch.squeeze(loss_fcn1.forward(gen_img[0], self.targ_img))
                #loss += 80 * - ssim_loss(gen_img, torch.unsqueeze(self.targ_img, dim=0))

                if loss < mse_min:
                    mse_min = loss
                    best_w = to_synt[0, i].detach().clone()
                    best_img = img

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                loss_tracker.append(loss.detach().cpu())

            img = self.G.synthesis(ws, noise_mode='const', force_fp32=True)
            im = self.genToPng(img)
            im.save(str(i) + '.png')

            plt.plot(loss_tracker)
            plt.show()
            ws[0, i] = best_w

        img = self.G.synthesis(ws, noise_mode='const', force_fp32=True)
        im = self.genToPng(img)
        im.save('best_proj_inv.png')

        return ws, best_img

    def recon_useCone(self, targ_path, saveVideo = True):
        # ---start---

        #mp4 stuff
        og_image = Image.open(targ_path).convert('RGB')
        og_image = og_image.resize((256, 256), resample=Image.Resampling.NEAREST)
        #w, h = og_image.size
        #s = min(w, h)
        #target_pil = og_image.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        #target_pil = target_pil.resize((self.G.img_resolution, self.G.img_resolution), Image.LANCZOS)
        #target_uint8 = np.array(target_pil, dtype=np.uint8)
        #ending getting target image for mp4

        self.engine.getVisuals(self.retinaPath, self.home_dir + '/' + targ_path, nargout = 0)
        og_visual = Image.open(self.visualPath).convert('RGB')

        top = get_concat_h_multi_blank([og_image, og_visual])
        top.save('combo.png')
        linear_image = torch.tensor(np.asarray(self.engine.getImageLinear(targ_path, self.retinaPath)),
                                    dtype=torch.float32, device="cuda")
        flat = torch.flatten(linear_image)
        coneExc = torch.matmul(self.render, flat)


        # ---reconstruction---
        best_w, imgs, visuals = self.useCone_step1(coneExc)

        ws, imgs1, visuals1 = self.layer_useCone(best_w, coneExc)
#
        bottoms = []
        for img, visual in zip(imgs, visuals):
            bottoms.append(get_concat_h_multi_blank([img, visual]))
        for img, visual in zip(imgs1, visuals1):
            bottoms.append(get_concat_h_multi_blank([img, visual]))
#

        if saveVideo:
            video = imageio.get_writer('proj_test.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
            print('Saving optimization progress video proj.mp4')
            top = np.array(top)
            for img in bottoms:

                img1 = np.array(img)
                #print(top.shape)

                #print(img1.shape)
                video.append_data(np.concatenate([top, img1], axis=0))
            video.close()

        return 0

    def useCone_step1(self, targ_coneExc, w_avg_samples = 10000, initial_learning_rate = 0.05):
        counter = 0
        visuals = []
        imgs = []

        z_samples = np.random.RandomState(123).randn(w_avg_samples, self.G.z_dim)
        w_samples = self.G.mapping(torch.from_numpy(z_samples).to("cuda"), None)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)

        w_opt = torch.tensor(w_avg, dtype=torch.float32, device="cuda", requires_grad=True)
        optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)

        loss_fcn = nn.MSELoss()
        # loss_fcn1 = lpips.LPIPS(net ='alex')
        # loss_fcn1.cuda()
        # ssim_loss = pytorch_ssim.SSIM()
        mse_min = np.inf

        loss_tracker = []

        for step in range(100):
            ws = w_opt.repeat([1, self.G.mapping.num_ws, 1])

            img = self.G.synthesis(ws, noise_mode='const', force_fp32=True)

            gen_png = self.genToPng(img)
            path = 'current_guess.png'
            gen_png.save(path)



            # ---start---
            linear_image = torch.tensor(np.asarray(self.engine.getImageLinear(path, self.retinaPath)),
                                        dtype=torch.float32, device="cuda")

            #here linear_image is the adjustment we need to make
            img = torch.squeeze(img)
            img = img.permute((1,2,0))
            # want to adjust img to gen_rec without loosing the comp map
            # img = linear + c
            # c = img - linear
            # img - c = linear

            c = img.detach().clone() - linear_image.detach().clone()
            img -= c

            flat = torch.flatten(img)
            gen_coneExc = torch.matmul(self.render, flat)


            # for MSELoss
            loss = loss_fcn(gen_coneExc, targ_coneExc)
            # loss += torch.squeeze(loss_fcn1.forward(gen_img[0], self.targ_img))
            # loss = - ssim_loss(gen_img, torch.unsqueeze(self.targ_img, dim = 0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_tracker.append(loss.detach().cpu())
            if loss < mse_min:
                counter +=1
                if counter < 50 or counter%4 == 0:
                    bigImage = gen_png.resize((256, 256), resample=Image.Resampling.NEAREST)
                    b_path = 'upscaled.png'
                    bigImage.save(b_path)
                    self.engine.getVisuals(self.retinaPath, self.home_dir + '/' + b_path, nargout = 0)
                    visuals.append(Image.open(self.visualPath).convert('RGB'))
                    imgs.append(bigImage)
                mse_min = loss
                best_w = ws

        img = self.G.synthesis(best_w, noise_mode='const', force_fp32=True)
        im = self.genToPng(img)
        im.save('step1_cone.png')

        plt.plot(loss_tracker)
        plt.show()

        return best_w, imgs, visuals

    def layer_useCone(self, ws, targ_coneExc):
        counter = 0
        visuals = []
        imgs = []
        loss_fcn = nn.MSELoss()
        #loss_fcn1 = lpips.LPIPS(net='alex')
        #loss_fcn1.cuda()
        #ssim_loss = pytorch_ssim.SSIM()
        mse_min = np.inf
        num_steps = 550
        ws = ws.detach().clone()

        for i in range(0, ws.shape[1]-1):
            loss_tracker = []

            w_opt = torch.tensor(ws[0, i], dtype=torch.float32, device="cuda", requires_grad=True)

            optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=0.05)

            beg = torch.unsqueeze(ws[0, :i], dim=0)
            end = torch.unsqueeze(ws[0, i + 1:], dim=0)

            for step in range(num_steps):
                mid = torch.unsqueeze(torch.unsqueeze(w_opt, dim=0), dim=0)
                to_synt = torch.cat((beg, mid, end), dim=1)

                img = self.G.synthesis(to_synt, noise_mode='const')
                gen_png = self.genToPng(img)
                path = 'current_guess.png'
                gen_png.save(path)


                # ---start---
                linear_image = torch.tensor(np.asarray(self.engine.getImageLinear(path, self.retinaPath)),
                                            dtype=torch.float32, device="cuda")

                # here linear_image is the adjustment we need to make
                img = torch.squeeze(img)
                img = img.permute((1, 2, 0))
                # want to adjust img to gen_rec without loosing the comp map
                # img = linear + c
                # c = img - linear
                # img - c = linear

                c = img.detach().clone() - linear_image.detach().clone()
                img -= c

                flat = torch.flatten(img)
                gen_coneExc = torch.matmul(self.render, flat)

                # for MSELoss
                loss = loss_fcn(gen_coneExc, targ_coneExc)
                #loss += 1.2 * torch.squeeze(loss_fcn1.forward(gen_img[0], self.targ_img))
                #loss += 80 * - ssim_loss(gen_img, torch.unsqueeze(self.targ_img, dim=0))

                if loss < mse_min:
                    counter += 1
                    if counter % 5 == 0:
                        bigImage = gen_png.resize((256, 256), resample=Image.Resampling.NEAREST)
                        b_path = 'upscaled.png'
                        bigImage.save(b_path)
                        self.engine.getVisuals(self.retinaPath, self.home_dir + '/' + b_path, nargout=0)
                        visuals.append(Image.open(self.visualPath).convert('RGB'))
                        imgs.append(bigImage)

                    mse_min = loss
                    best_w = to_synt[0, i].detach().clone()
                    best_img = img

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                loss_tracker.append(loss.detach().cpu())

            img = self.G.synthesis(ws, noise_mode='const', force_fp32=True)
            im = self.genToPng(img)
            im.save(str(i) + '.png')

            plt.plot(loss_tracker)
            plt.show()
            ws[0, i] = best_w

        img = self.G.synthesis(ws, noise_mode='const', force_fp32=True)
        im = self.genToPng(img)
        im.save('best_proj_cone.png')

        return ws, imgs, visuals


    def cone_blitz(self, targ_path, save_vid = True, w_avg_samples=10000, initial_learning_rate=0.05):


        linear_image = torch.tensor(np.asarray(self.engine.getImageLinear(targ_path, self.retinaPath)),
                                    dtype=torch.float32, device="cuda")
        flat = torch.flatten(linear_image)
        targ_coneExc = torch.matmul(self.render, flat)


        #Harrison - Refer to google doc
        counter = 0
        visuals = []
        imgs = []

        z_samples = np.random.RandomState(123).randn(w_avg_samples, self.G.z_dim)
        w_samples = self.G.mapping(torch.from_numpy(z_samples).to("cuda"), None)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)


        loss_fcn = nn.MSELoss()
        mse_min = np.inf
        num_steps = 50

        ws = w_avg.repeat([1, self.G.mapping.num_ws, 1])


        print(ws.shape)
        for i in range(1, ws.shape[1] - 1):
            print(i)
            loss_tracker = []

            w_opt = torch.tensor(ws[0, :i], dtype=torch.float32, device="cuda", requires_grad=True)

            optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=0.05)
            static = torch.unsqueeze(ws[0, i:], dim=0)

            for step in range(num_steps):
                opt = torch.unsqueeze(torch.unsqueeze(w_opt, dim=0), dim=0)
                to_synt = torch.cat((opt, static), dim=1)

                img = self.G.synthesis(to_synt, noise_mode='const')
                gen_png = self.genToPng(img)
                path = 'current_guess.png'
                gen_png.save(path)

                # ---start---
                linear_image = torch.tensor(np.asarray(self.engine.getImageLinear(path, self.retinaPath)),
                                            dtype=torch.float32, device="cuda")

                img = torch.squeeze(img)
                img = img.permute((1, 2, 0))

                c = img.detach().clone() - linear_image.detach().clone()
                img -= c

                flat = torch.flatten(img)
                gen_coneExc = torch.matmul(self.render, flat)

                loss = loss_fcn(gen_coneExc, targ_coneExc)

                if loss < mse_min:
                    counter += 1
                    if counter % 5 == 0 and save_vid:
                        bigImage = gen_png.resize((256, 256), resample=Image.Resampling.NEAREST)
                        b_path = 'upscaled.png'
                        bigImage.save(b_path)
                        self.engine.getVisuals(self.retinaPath, self.home_dir + '/' + b_path, nargout=0)
                        visuals.append(Image.open(self.visualPath).convert('RGB'))
                        imgs.append(bigImage)

                    mse_min = loss
                    best_w = to_synt[0, :i].detach().clone()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_tracker.append(loss.detach().cpu())

            plt.plot(loss_tracker)
            plt.show()
            ws[0, :i] = best_w


        #### Now the same but in reverse
        for i in range(0, ws.shape[1] - 1):
            loss_tracker = []

            w_opt = torch.tensor(ws[0, i:], dtype=torch.float32, device="cuda", requires_grad=True)

            optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=0.05)
            static = torch.unsqueeze(ws[0, :i], dim=0)

            for step in range(num_steps):
                opt = torch.unsqueeze(torch.unsqueeze(w_opt, dim=0), dim=0)
                to_synt = torch.cat((opt, static), dim=1)

                img = self.G.synthesis(to_synt, noise_mode='const')
                gen_png = self.genToPng(img)
                path = 'current_guess.png'
                gen_png.save(path)

                # ---start---
                linear_image = torch.tensor(np.asarray(self.engine.getImageLinear(path, self.retinaPath)),
                                            dtype=torch.float32, device="cuda")

                img = torch.squeeze(img)
                img = img.permute((1, 2, 0))

                c = img.detach().clone() - linear_image.detach().clone()
                img -= c

                flat = torch.flatten(img)
                gen_coneExc = torch.matmul(self.render, flat)

                loss = loss_fcn(gen_coneExc, targ_coneExc)

                if loss < mse_min:
                    counter += 1
                    if counter % 5 == 0 and save_vid:
                        bigImage = gen_png.resize((256, 256), resample=Image.Resampling.NEAREST)
                        b_path = 'upscaled.png'
                        bigImage.save(b_path)
                        self.engine.getVisuals(self.retinaPath, self.home_dir + '/' + b_path, nargout=0)
                        visuals.append(Image.open(self.visualPath).convert('RGB'))
                        imgs.append(bigImage)

                    mse_min = loss
                    best_w = to_synt[0, i:].detach().clone()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_tracker.append(loss.detach().cpu())

            plt.plot(loss_tracker)
            plt.show()
            ws[0, i:] = best_w


        img = self.G.synthesis(ws, noise_mode='const', force_fp32=True)
        im = self.genToPng(img)
        im.save('best_blitz_cone.png')

        return ws, imgs, visuals

    def cone_blitzv2(self, targ_coneExc, w_avg_samples=10000, initial_learning_rate=0.05):
        # Harrison - Refer to google doc
        counter = 0
        visuals = []
        imgs = []

        z_samples = np.random.RandomState(123).randn(w_avg_samples, self.G.z_dim)
        w_samples = self.G.mapping(torch.from_numpy(z_samples).to("cuda"), None)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)

        loss_fcn = nn.MSELoss()
        mse_min = np.inf
        num_steps = 50
        ws = w_avg.detach().clone()
        tracker = []

        for i in range(0, ws.shape[1] - 1):
            tracker.append(0)

        for i in range(0, ws.shape[1] - 1):
            tracker[i] = 1
            for j in range(len(tracker)):
                if tracker[j] == 1:
                    ws = self.cone_blitzv2Inner(ws, j, targ_coneExc)

        for i in range(0, ws.shape[1] - 1):
            tracker[i] = 0
            for j in range(len(tracker)):
                if tracker[j] == 1:
                    ws = self.cone_blitzv2Inner(ws, j, targ_coneExc)

        img = self.G.synthesis(ws, noise_mode='const', force_fp32=True)
        im = self.genToPng(img)
        im.save('best_coneblitzv2.png')

        return 0

    def cone_blitzv2Inner(self, ws, i, targ_coneExc):
        visuals = []
        imgs = []
        counter = 0
        num_steps = 50
        loss_fcn = nn.MSELoss()

        ws = ws.detach().clone()

        loss_tracker = []

        w_opt = torch.tensor(ws[0, i], dtype=torch.float32, device="cuda", requires_grad=True)

        optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=0.05)

        beg = torch.unsqueeze(ws[0, :i], dim=0)
        end = torch.unsqueeze(ws[0, i + 1:], dim=0)

        for step in range(num_steps):
            mid = torch.unsqueeze(torch.unsqueeze(w_opt, dim=0), dim=0)
            to_synt = torch.cat((beg, mid, end), dim=1)

            img = self.G.synthesis(to_synt, noise_mode='const')
            gen_png = self.genToPng(img)
            path = 'current_guess.png'
            gen_png.save(path)

            # ---start---
            linear_image = torch.tensor(np.asarray(self.engine.getImageLinear(path, self.retinaPath)),
                                        dtype=torch.float32, device="cuda")

            # here linear_image is the adjustment we need to make
            img = torch.squeeze(img)
            img = img.permute((1, 2, 0))
            # want to adjust img to gen_rec without loosing the comp map
            # img = linear + c
            # c = img - linear
            # img - c = linear

            c = img.detach().clone() - linear_image.detach().clone()
            img -= c

            flat = torch.flatten(img)
            gen_coneExc = torch.matmul(self.render, flat)

            # for MSELoss
            loss = loss_fcn(gen_coneExc, targ_coneExc)


            if loss < mse_min:
                counter += 1
                if counter % 5 == 0:
                    bigImage = gen_png.resize((256, 256), resample=Image.Resampling.NEAREST)
                    b_path = 'upscaled.png'
                    bigImage.save(b_path)
                    self.engine.getVisuals(self.retinaPath, self.home_dir + '/' + b_path, nargout=0)
                    visuals.append(Image.open(self.visualPath).convert('RGB'))
                    imgs.append(bigImage)

                mse_min = loss
                best_w = to_synt[0, i].detach().clone()
                best_img = img

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_tracker.append(loss.detach().cpu())

        img = self.G.synthesis(ws, noise_mode='const', force_fp32=True)
        im = self.genToPng(img)
        im.save(str(i) + '.png')

        plt.plot(loss_tracker)
        plt.show()
        ws[0, i] = best_w

        img = self.G.synthesis(ws, noise_mode='const', force_fp32=True)
        im = self.genToPng(img)
        im.save('best_proj_cone.png')

        return ws, visuals, imgs



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
		# Defines the learning rate, don't remember this from paper but seems good enough
        lr_ramp = min(1, (1 - t) / rampdown)
        lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
        lr_ramp = lr_ramp * min(1, t / rampup)
        return initial_lr * lr_ramp


    def invert_(self, start_layer, target_exc, steps, index, end_block_res):
        #keep
        learning_rate = self.config['lr'][index]
        print(f"Running round {index + 1} / {len(self.config['steps'])} of ILO.")


        #BEGINNING STUFF TO WORK THROUGH

        with torch.no_grad():
            if start_layer == 0:
                # var_list is the paramaters Adam optimizes over
                var_list = [self.z_hat_k]
                self.int_zs[None] #intermediate z's
            else:
                # TODO: what are gen_outs?
                #They are intermediate projections... Is this the proj back step?
                #Takes care of the function comp
                self.int_zs[-1].requires_grad = True
                # var_list is the paramaters Adam optimizes over
                var_list = [self.z_hat_k] + [self.int_zs[-1]]
                prev_gen_out = torch.ones(self.gen_outs[-1].shape, device=self.gen_outs[-1].device) * self.gen_outs[-1]

            # These can be used for the mapping back I think
            prev_latent = torch.ones(self.z_hat_k.shape, device=self.z_hat_k.device) * self.z_hat_k


            # set network that we will be optimizing over
            self.gen.start_layer = start_layer
            self.gen.end_layer = self.config['end_layer']


        optimizer = optim.Adam(var_list, lr=learning_rate)
        ps = SphericalOptimizer([self.latent_z])


        self.current_step += steps

            # 6.2 Set losses

        mse_min = np.inf
        mse_loss = 0
        p_loss = 0


            # 6.3 Start optimizing step by step
        pbar = tqdm(range(steps))
        for i in pbar:  # aka for each step
            loss = 0
            t = i / steps

            lr = self.get_lr(t, learning_rate)
            optimizer.param_groups[0]["lr"] = lr


            #latent_w = self.G.mapping(self.latent_z, None)
            #img_gen= self.G.synthesis(latent_w, noise_mode ='const')

            block_ws = []
            with torch.autograd.profiler.record_function('split_ws'):
                nn.misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
                ws = ws.to(torch.float32)
                w_idx = 0
                for res in self.block_resolutions:  # up to certain layer
                    block = getattr(self, f'b{res}')
                    block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                    w_idx += block.num_conv

            x = img = None
            i = 0
            for res, cur_ws in zip(self.block_resolutions, block_ws):
                if res == end_block_res:
                    break
                else:
                    block = getattr(self, f'b{res}')
                    int_z, int_gen_img = block(x, img, cur_ws, {})
                self.int_zs.append(int_z)

            #int_cone_exc = ISETBIO(int_gen_img)
            int_cone_exc = int_gen_img

            loss_fcn = nn.MSELoss()

            loss = loss_fcn(int_cone_exc, target_exc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # This would be where the ISETBio stuff comes in, right after this

            # 6.4  Create ISETBio cone excitations for the gen image


            # 6.5 Loss function time




            # compute loss
            #diff = torch.abs(cone_gen - self.original_imgs) - self.config['dead_zone_linear_alpha']
            #loss += self.config['dead_zone_linear'][index] * torch.max(torch.zeros(diff.shape, device=diff.device),
            #                                                           diff).mean()
#
            #mse_loss = F.mse_loss(cone_gen, self.original_imgs)
#
            #loss += self.config['mse'][index] * mse_loss
            #if self.config['pe'][index] != 0:
            #    if self.config['lpips_method'] == 'mask':
            #        p_loss = self.percept(self.downsampler_image_256(masked),
            #                              self.downsampler_image_256(self.original_imgs)).mean()
            #    elif self.config['lpips_method'] == 'fill':
            #        filled = mask * self.original_imgs + (1 - mask) * downsampled
            #        p_loss = self.percept(self.downsampler_1024_256(img_gen), self.downsampler_image_256(filled)).mean()
            #    else:
            #        raise NotImplementdError('LPIPS policy not implemented')
#
            #loss += self.config['pe'][index] * p_loss
#
            #loss += self.config['geocross'] * loss_geocross(self.latent_z[2 * start_layer:])
#
#
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()
#
            if self.project:
                # Step the sphericalOptimizer TODO: what does this do
                ps.step()

            # WILL PROBABLY WANT TO MAKE THESE TRUE, Pg 14 in paper says they do these
            if start_layer != 0 and self.config['do_project_gen_out']:
                deviation = project_onto_l1_ball(self.gen_outs[-1] - prev_gen_out,
                                                 self.config['max_radius_gen_out'][index])
                var_list[-1].data = (prev_gen_out + deviation).data
            if start_layer != 0 and self.config['do_project_latent']:
                deviation = project_onto_l1_ball(self.z_hat_k - prev_latent,
                                                 self.config['max_radius_latent'][index])
                var_list[0].data = (prev_latent + deviation).data

            if mse_loss < mse_min:
                mse_min = mse_loss
                self.best = int_gen_img.detach().cpu()

            pbar.set_description(
                (
                    f"perceptual: {p_loss:.4f};"
                    f" mse: {mse_loss:.4f};"
                )
            )

        # 8. Get optimal z  and img? for the layer



        ##LEFT OVER











        # TODO: check what happens when we are in the last layer
	  # No longer in the for loop, this only happens once per starting layer
      # with torch.no_grad():
      #     latent_w = self.mpl(self.latent_z)
      #     self.gen.end_layer = self.gen.start_layer
      #     intermediate_out, _  = self.gen([latent_w],
      #                                      input_is_latent=True,
      #                                      noise=self.noises,
      #                                      layer_in=self.gen_outs[-1],
      #                                      skip=None)
      #     self.gen_outs.append(intermediate_out)
      #     self.gen.end_layer = self.config['end_layer']

        #Driver for the whole thing, this used to also be called invert

    def step1(self, target_exc, num_steps = 1000, w_avg_samples = 10000,initial_learning_rate = 0.1):
        z_init = torch.randn([1, self.G.z_dim], dtype = torch.float32, device = "cuda", requires_grad=True).cuda()
        print(type(z_init))
        optimizer = torch.optim.Adam([z_init], lr = initial_learning_rate)
        loss_fcn = nn.MSELoss()

        for step in range(num_steps):
            gen_img = self.G(z_init, c=None, noise_mode='const')

            gen_img = (gen_img * 127.5 + 128).clamp(0, 255)

            #gen_exc = ISETBio[]
            gen_exc = gen_img

            loss = loss_fcn(gen_exc[0], target_exc)
            print('step: ', step, ', loss: ', loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        z_hat_k = z_init

        return z_hat_k, gen_img


    def reconstruct(self, target_image):
        print('Running with the following config....')
        pretty(self.config)

        #GET CONE EXCITATIONS FROM ISETBIO
        #target_exc = []
        #TO TEST WITH NO EXC
        target_exc = target_image


        #Get z_hat, step one in Algo1 ILO Paper
        self.z_hat_k, img2 = self.step1(target_exc)

        print('Saving image')

        img = (img2.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save('test3.png')


        #Replace with block resolution
        res_lst = self.G.synthesis.block_resolutions

        for i, steps in enumerate(self.config['steps']):
            #is is current layer, steps is steps per layer

            begin_from = i + self.config['start_layer']
            if begin_from > self.config['end_layer']:
                raise Exception('Attempting to go after end layer...')
            self.invert_(begin_from, target_exc, int(steps), i, res_lst[i])

        return target_image, (self.latent_z, self.noises, self.gen_outs), self.best

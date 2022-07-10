device = 'cuda'
import mmcv
import sys
sys.path.insert(0, '/home/anhvth8/gitprojects/stylegan2-ada-pytorch')
import dnnlib
import legacy
from torch_utils import misc
G_init_dict = mmcv.load('/home/anhvth8/gitprojects/stylegan2-ada-pytorch/configs/ffhq256_g_config.pkl')
G = dnnlib.util.construct_class_by_name(**G_init_dict['G_kwargs'], 
    **G_init_dict['common_kwargs']).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
with dnnlib.util.open_url('pretrained/ir_face.pkl') as f:
    resume_data = legacy.load_network_pkl(f)
for name, module in [('G_ema', G)]:
    misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

import torch

bz = 1
num_bz = 10000
from avcv.all import *
out_dir = './datasets/generated_ir_face'

def to_file(inp):
    img, i = inp
    out_file = osp.join(out_dir, f'{i:06d}.jpg')
    mmcv.imwrite(img, out_file)

for i in tqdm(range(num_bz)):
    z = torch.randn(bz,512).to(device)
    imgs = G(z, c=None)
    # tensor2imgs(imgs, 'bchw', mean=(127.5,127.5,127.5), )
    imgs = (imgs + 1) / 2
    imgs = torch.clip(imgs, 0, 1)*255
    imgs = imgs.byte().permute(0,2,3,1).cpu().numpy()

    inps = [(img, i+j)  for j, img in enumerate(imgs)]
    multi_thread(to_file, inps, verbose=False)
    # break
    
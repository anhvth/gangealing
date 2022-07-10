
from avcv.all import *
from models import get_stn
from utils.download import download_model, PRETRAINED_TEST_HYPERPARAMS
from utils.vis_tools.helpers import load_pil, save_image
from utils.distributed import setup_distributed, primary, get_rank, all_gatherv, synchronize, get_world_size
model_class = 'ir_face'  
resolution = 256  
import torch

setup_distributed()
def load_pil(path, resolution=None):
    arr = mmcv.imread(path, 0)
    arr = np.stack([arr]*3, -1)
    img = Image.fromarray(arr)
    if resolution is not None:
        img = img.resize((resolution, resolution), Image.LANCZOS)
    img = torch.tensor(np.asarray(img), device='cpu', dtype=torch.float).unsqueeze_(0).permute(0, 3, 1, 2)
    img = img.div(255.0).add(-0.5).mul(2)  
    return img  
from models import total_variation_loss
@torch.inference_mode()
def compute_flow_scores(batch, t):
    batch = batch.to('cuda')
    
    _, flows = t(batch, return_flow=True, iters=1, padding_mode='border')
    smoothness = total_variation_loss(flows, reduce_batch=False)
    return smoothness
def unnorm(img):
    img = (img-img.min())/(img.max()-img.min())
    img = img*255
    img = img.permute([1,2,0]).cpu().numpy().astype('uint8')
    return img
    
ckpt = download_model(model_class)  
stn = get_stn(['similarity', 'flow'], flow_size=128, supersize=resolution, num_heads=1).to('cuda')  
stn.load_state_dict(ckpt['t_ema'])  
test_kwargs = PRETRAINED_TEST_HYPERPARAMS[model_class]  
class DS:
    def __init__(self, paths, out_paths):
        self.paths = paths
        self.out_paths = out_paths
    def __getitem__(self, idx):
        return load_pil(self.paths[idx], resolution)[0], self.out_paths[idx]
    def __len__(self):
        return len(self.paths)
import torch
meta_path = '/home/anhvth8/data/eyestate-datasset/rldd_vit_196k.csv'
df = pd.read_csv(meta_path, index_col=0)
def gen_congeal_path(path):
    assert 'croped_faces' in path
    return path.replace('/croped_faces/', '/croped_congeal_faces/')
df['exists'] = df.path.apply(osp.exists)
df = df[df['exists']]
df['congeal_path'] = df.path.apply(gen_congeal_path)
def infer(dl):
    outs = []
    scores = []
    with torch.no_grad():
        pbar = tqdm(dl, total=len(dl))
        for input_img, out_path in pbar:
            aligned_img = stn.forward(input_img.cuda(), output_resolution=resolution, **test_kwargs)  
            aligned_img = [unnorm(_) for _ in aligned_img]
            for ali, op in zip(aligned_img, out_path):
                mmcv.imwrite(ali, op)
                outs.append(op)
            scores.extend((-compute_flow_scores(input_img, stn)).cpu().numpy().tolist())
    return outs, scores


n = len(df)
from loguru import logger

rank = get_rank()
world_size = get_world_size()
df = df[:n]
df = df[rank::world_size]
print(f'{rank=}, {df.head()=}')
ds = DS(df.path.tolist(), df.congeal_path.tolist())
dl = torch.utils.data.DataLoader(ds, 20, num_workers=10)

outs, scores = infer(dl)
print(len(outs), f'{rank=}, {len(ds)=}, {len(df)=}, {world_size=}')
mmcv.dump([outs, scores], f'/tmp/out_{rank}.pkl')

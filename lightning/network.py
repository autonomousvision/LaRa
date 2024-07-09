import torch,timm,random
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from lightning.renderer_2dgs import Renderer
from lightning.utils import MiniCam
from tools.rsh import rsh_cart_3

import pytorch_lightning as L
from torchvision import transforms


class DinoWrapper(L.LightningModule):
    """
    Dino v1 wrapper using huggingface transformer implementation.
    """
    def __init__(self, model_name: str, is_train: bool = False):
        super().__init__()
        self.model, self.processor = self._build_dino(model_name)
        self.freeze(is_train)

    def forward(self, image):
        # image: [N, C, H, W], on cpu
        # RGB image with [0,1] scale and properly size
        # This resampling of positional embedding uses bicubic interpolation
        outputs = self.model.forward_features(self.processor(image))

        return outputs[:,1:]
    
    def freeze(self, is_train:bool = False):
        print(f"======== image encoder is_train: {is_train} ========")
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = is_train

    @staticmethod
    def _build_dino(model_name: str, proxy_error_retries: int = 3, proxy_error_cooldown: int = 5):
        import requests
        try:
            model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True)
            data_config = timm.data.resolve_model_data_config(model)
            processor = transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
            return model, processor
        except requests.exceptions.ProxyError as err:
            if proxy_error_retries > 0:
                print(f"Huggingface ProxyError: Retrying in {proxy_error_cooldown} seconds...")
                import time
                time.sleep(proxy_error_cooldown)
                return DinoWrapper._build_dino(model_name, proxy_error_retries - 1, proxy_error_cooldown)
            else:
                raise err

class GroupAttBlock(L.LightningModule):
    def __init__(self, inner_dim: int, cond_dim: int, 
                 num_heads: int, eps: float,
                attn_drop: float = 0., attn_bias: bool = False,
                mlp_ratio: float = 2., mlp_drop: float = 0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(inner_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)

        self.cnn = nn.Conv3d(inner_dim, inner_dim, kernel_size=3, padding=1, bias=False)

        self.norm2 = norm_layer(inner_dim)
        self.norm3 = norm_layer(inner_dim)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )
        
    def forward(self, x, cond, group_axis, block_size):
        # x: [B, C, D, H, W]
        # cond: [B, L_cond, D_cond]

        B,C,D,H,W = x.shape

        # Unfold the tensor into patches
        patches = x.unfold(2, block_size, block_size).unfold(3, block_size, block_size).unfold(4, block_size, block_size)
        patches = patches.reshape(B, C, -1, block_size**3)
        patches = torch.einsum('bcgl->bglc',patches).reshape(B*group_axis**3, block_size**3,C)
     
        # cross attention
        patches = patches + self.cross_attn(self.norm1(patches), cond, cond, need_weights=False)[0]
        patches = patches + self.mlp(self.norm2(patches))

        # 3D CNN
        patches = self.norm3(patches)
        patches = patches.view(B, group_axis,group_axis,group_axis,block_size,block_size,block_size,C) 
        patches = torch.einsum('bdhwzyxc->bcdzhywx',patches).reshape(x.shape)
        patches = patches + self.cnn(patches)

        return patches
    

class VolTransformer(L.LightningModule):
    """
    Transformer with condition and modulation that generates a triplane representation.
    
    Reference:
    Timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L486
    """
    def __init__(self, embed_dim: int, image_feat_dim: int, n_groups: list,
                 vol_low_res: int, vol_high_res: int, out_dim: int,
                 num_layers: int, num_heads: int,
                 eps: float = 1e-6):
        super().__init__()

        # attributes
        self.vol_low_res = vol_low_res
        self.vol_high_res = vol_high_res
        self.out_dim = out_dim
        self.n_groups = n_groups
        self.block_size = [vol_low_res//item for item in n_groups]
        self.embed_dim = embed_dim

        # modules
        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, vol_low_res,vol_low_res,vol_low_res) * (1. / embed_dim) ** 0.5)
        self.layers = nn.ModuleList([
            GroupAttBlock(
                inner_dim=embed_dim, cond_dim=image_feat_dim, num_heads=num_heads, eps=eps)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim, eps=eps)
        self.deconv = nn.ConvTranspose3d(embed_dim, out_dim, kernel_size=2, stride=2, padding=0)

    def forward(self, image_feats):
        # image_feats: [B, N_views, C, DHW]
        # camera_embeddings: [N, D_mod]
        
        B,V,C,D,H,W = image_feats.shape
        
        volume_feats = []
        for n_group in self.n_groups:
            block_size = D//n_group
            blocks = image_feats.unfold(3, block_size, block_size).unfold(4, block_size, block_size).unfold(5, block_size,block_size)
            blocks = blocks.contiguous().view(B,V,C,n_group**3,block_size**3)
            blocks = torch.einsum('bvcgl->bgvlc',blocks).reshape(B*n_group**3,block_size**3*V,C)
            volume_feats.append(blocks)

        x = self.pos_embed.repeat(B, 1,1,1,1)  # [N, G, L, D]

        for i, layer in enumerate(self.layers):
            group_idx = i%len(self.block_size)
            x = layer(x, volume_feats[group_idx], self.n_groups[group_idx], self.block_size[group_idx])

        x = self.norm(torch.einsum('bcdhw->bdhwc',x))
        x = torch.einsum('bdhwc->bcdhw',x)

        # separate each plane and apply deconv
        x_up = self.deconv(x)  # [3*N, D', H', W']
        x_up = torch.einsum('bcdhw->bdhwc',x_up).contiguous()
        return x_up


def get_pose_feat(src_exts, tar_ext, src_ixts, W, H):
    """
    src_exts: [B,N,4,4]
    tar_ext: [B,4,4]
    src_ixts: [B,N,3,3]
    """

    B = src_exts.shape[0]
    c2w_ref = src_exts[:,0].view(B,-1)
    normalize_facto = torch.tensor([W,H]).unsqueeze(0).to(c2w_ref)
    fx_fy = src_ixts[:,0,[0,1],[0,1]]/normalize_facto
    cx_cy = src_ixts[:,0,[0,1],[2,2]]/normalize_facto

    return torch.cat((c2w_ref,fx_fy,fx_fy), dim=-1)

def projection(grid, w2cs, ixts):

    points = grid.reshape(1,-1,3) @ w2cs[:,:3,:3].permute(0,2,1) + w2cs[:,:3,3][:,None]
    points = points @ ixts.permute(0,2,1)
    points_xy = points[...,:2]/points[...,-1:]
    return points_xy, points[...,-1:]


class ModLN(L.LightningModule):
    """
    Modulation with adaLN.
    
    References:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
    """
    def __init__(self, inner_dim: int, mod_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_dim, inner_dim * 2),
        )

    @staticmethod
    def modulate(x, shift, scale):
        # x: [N, L, D]
        # shift, scale: [N, D]
        return x * (1 + scale) + shift

    def forward(self, x, cond):
        shift, scale = self.mlp(cond).chunk(2, dim=-1)  # [N, D]
        return self.modulate(self.norm(x), shift, scale)  # [N, L, D]
    
class Decoder(L.LightningModule):
    def __init__(self, in_dim, sh_dim, scaling_dim, rotation_dim, opacity_dim, K=1, latent_dim=256):
        super(Decoder, self).__init__()

        self.K = K
        self.sh_dim = sh_dim
        self.opacity_dim = opacity_dim
        self.scaling_dim = scaling_dim
        self.rotation_dim  = rotation_dim
        self.out_dim = 3 + sh_dim + opacity_dim + scaling_dim + rotation_dim

        num_layer = 2
        layers_coarse = [nn.Linear(in_dim, in_dim), nn.ReLU()] + \
                 [nn.Linear(in_dim, in_dim), nn.ReLU()] * (num_layer-1) + \
                 [nn.Linear(in_dim, self.out_dim*K)]
        self.mlp_coarse = nn.Sequential(*layers_coarse)


        cond_dim = 8
        self.norm = nn.LayerNorm(in_dim)
        self.cross_att = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=8, kdim=cond_dim, vdim=cond_dim,
            dropout=0.0, bias=False, batch_first=True)
        layers_fine = [nn.Linear(in_dim, 64), nn.ReLU()] + \
                 [nn.Linear(64, self.sh_dim)]
        self.mlp_fine = nn.Sequential(*layers_fine)
        
        self.init(self.mlp_coarse)
        self.init(self.mlp_fine)

    def init(self, layers):
        # MLP initialization as in mipnerf360
        init_method = "xavier"
        if init_method:
            for layer in layers:
                if not isinstance(layer, torch.nn.Linear):
                    continue 
                if init_method == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(layer.weight.data)
                elif init_method == "xavier":
                    torch.nn.init.xavier_uniform_(layer.weight.data)
                torch.nn.init.zeros_(layer.bias.data)

    
    def forward_coarse(self, feats, opacity_shift, scaling_shift):
        parameters = self.mlp_coarse(feats).float()
        parameters = parameters.view(*parameters.shape[:-1],self.K,-1)
        offset, sh, opacity, scaling, rotation = torch.split(
            parameters, 
            [3, self.sh_dim, self.opacity_dim, self.scaling_dim, self.rotation_dim],
            dim=-1
            )
        opacity = opacity + opacity_shift 
        scaling = scaling + scaling_shift 
        offset = torch.sigmoid(offset)*2-1.0

        B = opacity.shape[0]
        sh = sh.view(B,-1,self.sh_dim//3,3)
        opacity = opacity.view(B,-1,self.opacity_dim)
        scaling = scaling.view(B,-1,self.scaling_dim)
        rotation = rotation.view(B,-1,self.rotation_dim)
        offset = offset.view(B,-1,3)
        
        return offset, sh, scaling, rotation, opacity

    def forward_fine(self, volume_feat, point_feats):
        volume_feat = self.norm(volume_feat.unsqueeze(1))
        x = self.cross_att(volume_feat, point_feats, point_feats, need_weights=False)[0]
        sh = self.mlp_fine(x).float()
        return sh
    
class Network(L.LightningModule):
    def __init__(self, cfg, white_bkgd=True):
        super(Network, self).__init__()

        self.cfg = cfg
        self.scene_size = 0.5
        self.white_bkgd = white_bkgd

        # modules
        self.img_encoder = DinoWrapper(
            model_name=cfg.model.encoder_backbone,
            is_train=True,
        )
       
        encoder_feat_dim = self.img_encoder.model.num_features
        self.dir_norm = ModLN(encoder_feat_dim, 16*2, eps=1e-6)

        # build volume position
        self.grid_reso = cfg.model.vol_embedding_reso
        self.register_buffer("dense_grid", self.build_dense_grid(self.grid_reso))
        self.register_buffer("centers", self.build_dense_grid(self.grid_reso*2))

        # view embedding
        if cfg.model.view_embed_dim > 0:
            self.view_embed = nn.Parameter(torch.randn(1, 4, cfg.model.view_embed_dim,1,1,1) * (1. / cfg.model.view_embed_dim) ** 0.5)
        
        # build volume transformer
        self.n_groups = cfg.model.n_groups
        vol_embedding_dim = cfg.model.embedding_dim
        self.vol_decoder = VolTransformer(
            embed_dim=vol_embedding_dim, image_feat_dim=encoder_feat_dim+cfg.model.view_embed_dim,
            vol_low_res=self.grid_reso, vol_high_res=self.grid_reso*2, out_dim=cfg.model.vol_embedding_out_dim, n_groups=self.n_groups,
            num_layers=cfg.model.num_layers, num_heads=cfg.model.num_heads,
        )
        self.feat_vol_reso = cfg.model.vol_feat_reso
        self.register_buffer("volume_grid", self.build_dense_grid(self.feat_vol_reso))
        
        # grouping configuration
        self.n_offset_groups = cfg.model.n_offset_groups
        self.register_buffer("group_centers", self.build_dense_grid(self.grid_reso*2))
        self.group_centers = self.group_centers.reshape(1,-1,3)

        # 2DGS model
        self.sh_dim = (cfg.model.sh_degree+1)**2*3
        self.scaling_dim, self.rotation_dim = 2, 4
        self.opacity_dim = 1
        self.out_dim = self.sh_dim + self.scaling_dim + self.rotation_dim + self.opacity_dim

        self.K = cfg.model.K
        vol_embedding_out_dim = cfg.model.vol_embedding_out_dim
        self.decoder = Decoder(vol_embedding_out_dim, self.sh_dim, self.scaling_dim, self.rotation_dim, self.opacity_dim, self.K)
        self.gs_render = Renderer(sh_degree=cfg.model.sh_degree, white_background=white_bkgd, radius=1)

        # parameters initialization
        self.opacity_shift = -2.1792
        self.voxel_size = 2.0/(self.grid_reso*2)
        self.scaling_shift = np.log(0.5*self.voxel_size/3.0)
        

    def build_dense_grid(self, reso):
        array = torch.arange(reso, device=self.device)
        grid = torch.stack(torch.meshgrid(array, array, array, indexing='ij'),dim=-1)
        grid = (grid + 0.5) / reso * 2 -1
        return grid.reshape(reso,reso,reso,3)*self.scene_size

    
    def build_feat_vol(self, src_inps, img_feats, n_views_sel, batch):

        h,w = src_inps.shape[-2:]
        src_ixts = batch['tar_ixt'][:,:n_views_sel].reshape(-1,3,3)
        src_w2cs = batch['tar_w2c'][:,:n_views_sel].reshape(-1,4,4)
        

        img_wh = torch.tensor([w,h], device=self.device)
        point_img,_ = projection(self.volume_grid, src_w2cs, src_ixts) 
        point_img = (point_img+ 0.5)/img_wh*2 - 1.0
        
        # viewing direction
        rays = batch['tar_rays_down'][:,:n_views_sel]
        feats_dir = self.ray_to_plucker(rays).reshape(-1,*rays.shape[2:])
        feats_dir = torch.cat((rsh_cart_3(feats_dir[...,:3]),rsh_cart_3(feats_dir[...,3:6])),dim=-1)

        # query features
        img_feats =  torch.einsum('bchw->bhwc',img_feats)
        img_feats = self.dir_norm(img_feats, feats_dir)
        img_feats = torch.einsum('bhwc->bchw',img_feats)

        n_channel = img_feats.shape[1]
        feats_vol = F.grid_sample(img_feats.float(), point_img.unsqueeze(1), align_corners=False).to(img_feats)

        # img features
        feats_vol = feats_vol.view(-1,n_views_sel,n_channel,self.feat_vol_reso,self.feat_vol_reso,self.feat_vol_reso)

        return feats_vol
    
    def _check_mask(self, mask):
        ratio = torch.sum(mask)/np.prod(mask.shape)
        if ratio < 1e-3: 
            mask = mask + torch.rand(mask.shape, device=self.device)>0.8
        elif  ratio > 0.5 and self.training: 
            # avoid OMM
            mask = mask * torch.rand(mask.shape, device=self.device)>0.5
        return mask
            
    def get_point_feats(self, idx, img_ref, renderings, n_views_sel, batch, points, mask):
        
        points = points[mask]
        n_points = points.shape[0]
        
        h,w = img_ref.shape[-2:]
        src_ixts = batch['tar_ixt'][idx,:n_views_sel].reshape(-1,3,3)
        src_w2cs = batch['tar_w2c'][idx,:n_views_sel].reshape(-1,4,4)
        
        img_wh = torch.tensor([w,h], device=self.device)
        point_xy, point_z = projection(points, src_w2cs, src_ixts)
        point_xy = (point_xy + 0.5)/img_wh*2 - 1.0

        imgs_coarse = torch.cat((renderings['image'],renderings['acc_map'].unsqueeze(-1),renderings['depth']), dim=-1)
        imgs_coarse = torch.cat((img_ref, torch.einsum('bhwc->bchw', imgs_coarse)),dim=1)
        feats_coarse = F.grid_sample(imgs_coarse, point_xy.unsqueeze(1), align_corners=False).view(n_views_sel,-1,n_points).to(imgs_coarse)
        
        z_diff = (feats_coarse[:,-1:] - point_z.view(n_views_sel,-1,n_points)).abs()
                    
        point_feats = torch.cat((feats_coarse[:,:-1],z_diff), dim=1)#[...,_mask]
        
        return point_feats, mask
        
        
    def ray_to_plucker(self, rays):
        origin, direction = rays[...,:3], rays[...,3:6]
        # Normalize the direction vector to ensure it's a unit vector
        direction = F.normalize(direction, p=2.0, dim=-1)
        
        # Calculate the moment vector (M = O x D)
        moment = torch.cross(origin, direction, dim=-1)
        
        # Plucker coordinates are L (direction) and M (moment)
        return torch.cat((direction, moment),dim=-1)
    
    def get_offseted_pt(self, offset, K):
        B = offset.shape[0]
        half_cell_size = 0.5*self.scene_size/self.n_offset_groups
        centers = self.group_centers.unsqueeze(-2).expand(B,-1,K,-1).reshape(offset.shape) + offset*half_cell_size
        return centers
    
    def forward(self, batch, with_fine=False, return_buffer=False):
        
        B,N,H,W,C = batch['tar_rgb'].shape
        if self.training:
            n_views_sel = random.randint(2, 4) if self.cfg.train.use_rand_views else self.cfg.n_views
        else:
            n_views_sel = self.cfg.n_views

        _inps =batch['tar_rgb'][:,:n_views_sel].reshape(B*n_views_sel,H,W,C)
        _inps = torch.einsum('bhwc->bchw', _inps)

        # image encoder
        img_feats = torch.einsum('blc->bcl', self.img_encoder(_inps))
        token_size = int(np.sqrt(H*W/img_feats.shape[-1]))
        img_feats = img_feats.reshape(*img_feats.shape[:2],H//token_size,W//token_size)

        # build 3D volume
        feat_vol = self.build_feat_vol(_inps, img_feats, n_views_sel, batch) # B n_views_sel C D H W
        
        # view embedding
        if self.cfg.model.view_embed_dim > 0:
            feat_vol = torch.cat((feat_vol, self.view_embed[:,:n_views_sel].expand(B,-1,-1,self.feat_vol_reso,self.feat_vol_reso,self.feat_vol_reso)),dim=2)

        # decoding
        volume_feat_up = self.vol_decoder(feat_vol)

        # rendering
        _offset_coarse, _shs_coarse, _scaling_coarse, _rotation_coarse, _opacity_coarse = self.decoder.forward_coarse(volume_feat_up, self.opacity_shift, self.scaling_shift)

        # convert to local positions
        _centers_coarse = self.get_offseted_pt(_offset_coarse, self.K)


        _opacity_coarse_tmp = self.gs_render.opacity_activation(_opacity_coarse).squeeze(-1)
        masks =  _opacity_coarse_tmp > 0.005

        render_img_scale = batch.get('render_img_scale', 1.0)
        
        volume_feat_up = volume_feat_up.view(B,-1,volume_feat_up.shape[-1])
        _inps = _inps.reshape(B,n_views_sel,C,H,W).float()
        
        outputs,render_pkg = [],[]
        for i in range(B):
 
            znear, zfar = batch['near_far'][i]
            fovx,fovy = batch['fovx'][i], batch['fovy'][i]
            height, width = int(batch['meta']['tar_h'][i]*render_img_scale), int(batch['meta']['tar_w'][i]*render_img_scale)

            mask = masks[i].detach()

            _centers = _centers_coarse[i]
            if return_buffer:
                render_pkg.append((_centers, _shs_coarse[i], _opacity_coarse[i], _scaling_coarse[i], _rotation_coarse[i]))
            
            outputs_view = []
            tar_c2ws = batch['tar_c2w'][i]
            for j, c2w in enumerate(tar_c2ws):
                
                bg_color = batch['bg_color'][i,j]
                self.gs_render.set_bg_color(bg_color)
            
                cam = MiniCam(c2w, width, height, fovy, fovx, znear, zfar, self.device)
                rays_d = batch['tar_rays'][i,j]
                
                # coarse
                frame = self.gs_render.render_img(cam, rays_d, _centers, _shs_coarse[i], _opacity_coarse[i], _scaling_coarse[i], _rotation_coarse[i], self.device)
                outputs_view.append(frame)
                
            rendering_coarse = {k: torch.stack([d[k] for d in outputs_view[:n_views_sel]]) for k in outputs_view[0]}

            # fine
            if with_fine:
                    
                mask = self._check_mask(mask)
                point_feats, mask = self.get_point_feats(i, _inps[i], rendering_coarse, n_views_sel, batch, _centers, mask)
                
                    
                _centers = _centers[mask]
                point_feats =  torch.einsum('lcb->blc', point_feats)

                volume_point_feat = volume_feat_up[i].unsqueeze(1).expand(-1,self.K,-1)[mask.view(-1,self.K)]
                _shs_fine = self.decoder.forward_fine(volume_point_feat, point_feats).view(-1,*_shs_coarse.shape[-2:]) + _shs_coarse[i][mask]
                
                if return_buffer:
                    render_pkg.append((_centers, _shs_fine, _opacity_coarse[i], _scaling_coarse[i], _rotation_coarse[i], mask))
                    
                for j,c2w in enumerate(tar_c2ws):
                    
                    bg_color = batch['bg_color'][i,j]
                    self.gs_render.set_bg_color(bg_color)
                
                    rays_d = batch['tar_rays'][i,j]
                    cam = MiniCam(c2w, width, height, fovy, fovx, znear, zfar, self.device)
                    frame_fine = self.gs_render.render_img(cam, rays_d, _centers, _shs_fine, _opacity_coarse[i][mask], _scaling_coarse[i][mask], _rotation_coarse[i][mask], self.device, prex='_fine')
                    outputs_view[j].update(frame_fine)
            
            outputs.append({k: torch.cat([d[k] for d in outputs_view], dim=1) for k in outputs_view[0]})

        outputs = {k: torch.stack([d[k] for d in outputs]) for k in outputs[0]}
        if return_buffer:
            outputs.update({'render_pkg':render_pkg}) 
        return outputs


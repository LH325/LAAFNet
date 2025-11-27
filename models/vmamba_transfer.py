import torch.nn as nn
from timm.models.layers import DropPath,to_2tuple
import torch
from mamba_ssm.ops.triton.layernorm import RMSNorm,rms_norm_fn,layer_norm_fn
from torch import Tensor
from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba
from typing import Optional
from rope import *
from timm.models.layers import trunc_normal_,lecun_normal_
import random
from einops import rearrange

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224,patch_size=16,stride=16,in_channels=3,embed_dim=768,norm_layer=None,flatten=True):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)#将img_size和patch_size化成元组的形式
        self.img_size = img_size
        self.patch_size = patch_size
        #一个patch形成一个grid（网格），这里记录网格的形状
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1 , (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]#总共的patch个数
        self.flatten = flatten
        #打patch的操作，实际为卷积的操作(为了不重复卷积，步长的大小理论上因该等于卷积核的大小）
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()#nn.Identity的输入等于输出，通常作为占位层使用
 
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1],\
            f"Input img size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"
        x = self.proj(x)#B,C,H,W——>B,embed_dim,grid_size,grid_size
    
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)#B,embed_dim,grid_size,grid_size——>B,embed_dim,grid_size*grid_size——>B,grid_size*grid_size,embed_dim
      
        x = self.norm(x)
        
        return x
class Block(nn.Module):
    def __init__(self,dim,mixer_cls,norm_cls=nn.LayerNorm,fuse_add_norm=False,residual_in_fp32=False,drop_path=0):
        super(Block,self).__init__()
        self.fused_add_norm = fuse_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        self.drop_path = DropPath(drop_path)
        self.reidual_in_fp32 = residual_in_fp32
        if self.fused_add_norm:
            assert RMSNorm is not None,"RMSNorm fails"
            assert isinstance(self.norm,(nn.LayerNorm,RMSNorm))
    
    def forward(self,hidden_states:Tensor,residual:Optional[Tensor]=None,inference_params=None):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.reidual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm,RMSNorm) else layer_norm_fn

            if residual is None:
                hidden_states,residual = fused_add_norm_fn(hidden_states,self.norm.weight,self.norm.bias,residual=residual,
                                                    prenorm=True,residual_in_fp32=self.reidual_in_fp32,eps=self.norm.eps)
            else:
                hidden_states,residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm = True,
                    residual_in_fp32=self.reidual_in_fp32,
                    eps = self.norm.eps
                )
        hidden_states = self.mixer(hidden_states,inference_params=inference_params)
        return hidden_states,residual
def create_block(d_model,ssm_cfg=None,norm_epsilon=1e-5,drop_path=0.,rms_norm=False,residual_in_fp32=False,fused_add_norm=False,
                 layer_idx=None,device=None,dtype=None,if_bimamba=None,bimamba_type="none",if_devide_out=False,init_layer_scale=None):
    if if_bimamba:
        bimamba_type="v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs={"device":device,"dtype":dtype}
    mixer_cls = partial(Mamba,
                        layer_idx=layer_idx,
                        bimamba_type=bimamba_type,
                        if_devide_out=if_devide_out,
                        init_layer_scale = init_layer_scale,
                        **ssm_cfg,**factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm,eps=norm_epsilon,**factory_kwargs)
    block = Block(dim=d_model,mixer_cls=mixer_cls,norm_cls=norm_cls,drop_path=drop_path,fuse_add_norm=fused_add_norm,residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block

class VisionMamba(nn.Module):
    def __init__(self,image_size=224,patch_size=16,stride = 16,depth=24,embed_dim=192,channels=3,num_classes=1000,ssm_cfg=None,drop_rate=0.,drop_path_rate=0.1,
                 norm_epsilon:float=1e-5,rms_norm:bool=False,fused_add_norm=False,residual_in_fp32=False,device=None,dtype=None,pt_hw_seq_len=14,if_bidirectional=False,
                 final_pool_type='none',if_abs_pose_embed=False,if_rope = False,if_rope_residual = False,flip_img_sequeces_ratio=-1.,if_bimamba=False,bimamba_type="none",if_cls_token=False,if_devide_out=False,
                 init_layer_scale=None,use_double_cls_token=False,use_middle_cls_token=False,**kwargs):
       
        factory_kwargs = {"device":device,"dtype":dtype}
        kwargs.update(factory_kwargs)
        super(VisionMamba,self).__init__()
        self.residual_in_fp32 = residual_in_fp32   
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pose_embed
        self.if_rope= if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequeces_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = use_double_cls_token
        self.use_middle_cls_token = use_middle_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(img_size=image_size,patch_size=patch_size,stride=stride,in_channels=channels,embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # cls_token
        if if_cls_token:
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(torch.zeros(1,1,self.embed_dim))
                self.cls_token_tail = nn.Parameter(torch.zeros(1,1,self.embed_dim))
                self.num_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1,1,self.embed_dim))

        # position_embeding
        if if_abs_pose_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1,num_patches+self.num_tokens,self.embed_dim)) 
            self.pos_drop = nn.Dropout(p=drop_rate)
            

        # Rope
        if if_rope:
            half_head_dim = embed_dim//2
            hw_seq_len = image_size//patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim = half_head_dim,pt_seq_len = pt_hw_seq_len,ft_seq_len = hw_seq_len,
            )
        self.head = nn.Linear(self.num_features,num_classes) if num_classes>0 else nn.Identity()
        
        # 随机失活
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate>0. else nn.Identity()
        self.layers = nn.ModuleList([
            create_block(
                embed_dim,
                ssm_cfg=ssm_cfg,
                norm_epsilon=norm_epsilon,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                if_bimamba=if_bimamba,
                bimamba_type=bimamba_type,
                drop_path=inter_dpr[i],
                if_devide_out=if_devide_out,
                init_layer_scale=init_layer_scale,
                **factory_kwargs
            )
            for i in range(depth)
        ])
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim,eps=norm_epsilon,**factory_kwargs
        )
        if if_abs_pose_embed:
            trunc_normal_(self.pos_embed, std=.02)
        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head,std=.02)
                trunc_normal_(self.cls_token_tail,std=.02)
            else:
                trunc_normal_(self.cls_token,std=.02)
    def forward_features(self,x,inference_params=None,if_random_cls_token_position=False,if_random_token_rank=False):
        x = self.patch_embed(x)
        B, M, _ = x.shape

        if self.if_cls_token:
            if self.use_double_cls_token:
                cls_token_head = self.cls_token_head.expand(B,-1,-1)
                cls_token_tail = self.cls_token_tail.expand(B,-1,-1)

                token_position = [0,M+1]
                x = torch.cat((cls_token_head,x,cls_token_tail),dim=1)
                M = x.shape[1]
            else:
                if self.use_middle_cls_token:
                    cls_token = self.cls_token.expand(B,-1,-1)
                    token_position = M//2
                    x = torch.cat((x[:,:token_position,:],cls_token,x[:,token_position:,:]),dim=1)
                elif if_random_cls_token_position:
                    cls_token = self.cls_token.expand(B,-1,-1)
                    token_position = random.randint(0,M)
                    x = torch.cat((x[:,:token_position,:],cls_token,x[:,token_position:,:]),dim=1)
                    print("token_position",token_position)
                else:
                    cls_token = self.cls_token.expand(0,-1,-1)
                    token_position = 0
                    x = torch.cat((cls_token,x),dim=1)
                M = x.shape[1]
        if self.if_abs_pos_embed:
            x = x+self.pos_embed
            x = self.pos_drop(x)
        
        if if_random_token_rank:
            shuffle_indices  = torch.randperm(M)

            if isinstance(token_position,list):
                print("original value", x[0,token_position[0],0],x[0,token_position[1],0])
            else:
                print("original value",x[0,token_position,0])
            print("original token_position", token_position)

            x = x[:shuffle_indices,:]

            if isinstance(token_position,list):
                new_token_position = [torch.where(shuffle_indices == token_position[i])[0].item() for i in range(len(token_position))]
                token_position = new_token_position
            else:
                token_position = torch.where(shuffle_indices==token_position)[0].item()
            
            if isinstance(token_position,list):
                print("new value",x[0,token_position[0],0],x[0,token_position[1],0])
            else:
                print("new value",x[0,token_position,0])
            print("new token_position:",token_position)
        if_flip_img_sequences = False
        if self.flip_img_sequences_ratio >0 and (self.flip_img_sequence_ratio - random.random()) > 1e-5:
            x = x.flip([1])
            if_flip_img_sequences = True
        
        # mamba
        residual = None
        hidden_states = x
        if not self.if_bidirectional:
            for layer in self.layers:
                if if_flip_img_sequences and self.if_rope:
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])
                
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual = self.rope(residual)
                
                hidden_states,residual = layer(hidden_states,residual,inference_params = inference_params)
        else:
            for i in range(len(self.layers)//2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residual:
                        residual= self.rope(residual)
                hidden_states_f, residual_f = self.layers[i*2](
                    hidden_states,residual,inference_params=inference_params
                )
                hidden_states_b,residual_b = self.layer[i*2+1](
                    hidden_states.flip([1]),
                    None if residual ==None else residual.flip([1]),
                    inference_params = inference_params
                )
                hidden_states = hidden_states_f + hidden_states_b.flip([1])
                residual = residual_f + residual_b.flip([1])
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual +self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f,RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(self.drop_path(hidden_states),self.norm_f.weight,self.norm_f.bias,eps=self.norm_f.eps,residual=residual,
                                              residual_in_fp32=self.residual_in_fp32)
        if self.if_cls_token:
            if self.use_double_cls_token:
                return (hidden_states[:,token_position[0],:]+hidden_states[:,token_position[1],:])/2
            else:
                if self.use_middle_cls_token:
                    return hidden_states[:,token_position,:]
                elif if_random_cls_token_position:
                    return hidden_states[:,token_position,:]
                else:
                    return hidden_states[:,token_position,:]
        if self.final_pool_type == 'none':
            return hidden_states[:,-1,:]
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            return hidden_states
        elif self.final_pool_type == 'all':
            return hidden_states
        else:
            raise NotImplementedError

    def forward(self,x,return_features = True,inference_params=None,if_random_cls_token_position=False,if_random_token_rank=False):
        x = self.forward_features(x,inference_params,if_random_cls_token_position=if_random_cls_token_position,
                                  if_random_token_rank=if_random_token_rank)
     
        if return_features:
            B, C = x.shape
            x = x.expand([64, 64, B, C])
            x = rearrange(x,'w h b c -> b c w h')
            return x
        x = self.head(x)
  
        if self.final_pool_type == 'max':
            x = x.max(dim=1)[0]
     
        return x 
def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VisionMamba(image_size=512,
        patch_size=16,
        embed_dim=192,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pose_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="V2",
        if_cls_token=True,
        if_devide_out=True,
        use_middle_cls_token=False,
        use_double_cls_token=True
    ).to(device)
    return model
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = test()
    matrix = torch.randn((4,3,512,512)).to(device)
    preds = model(matrix)
    print(preds.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

device = 'cuda' if torch.cuda.is_available() else 'cpu'





#_______________________________________________________________________________


@dataclass
class Config:
    num_classes: int = 10
    img_size: int = 28
    im_channels: int = 1
    patch_size: int = 4

    n_head: int = 4
    n_layer: int = 4
    n_embd: int = 768

    dropout = 0.2

    @property
    def n_patch(self):
        return (self.img_size//self.patch_size)**2
      
      
@dataclass
class ViTBaseConfig:
    num_classes: int = 100        # Imagenet 100
    img_size: int = 224      
    im_channels: int = 3
    patch_size: int = 16

    n_head: int = 12
    n_layer: int = 12
    n_embd: int = 768

    dropout = 0.1  

    @property
    def n_patch(self):
        return (self.img_size//self.patch_size)**2

#_______________________________________________________________________________



class ViT(nn.Module):

    def __init__(self,config):
      super().__init__()


      self.config = config

      self.embd = PatchEmbedding(config)

      self.block = nn.ModuleList([Block(config)  for i in range(config.n_layer)])


      self.ln = nn.LayerNorm(config.n_embd)

      self.layer = nn.Linear(config.n_embd,config.num_classes)





    def forward(self,x,targets = None):

      B,C,H,W = x.shape


      #embedding
      out =  self.embd(x)

      #blocks
      for block in self.block:
        out = block(out)

      #layer norm
      out = self.ln(out)

      #linear layer
      out = self.layer(out[:,0])


      if targets is None:
        return out
      else:
        return out,F.cross_entropy(out,targets.view(-1))



#_______________________________________________________________________________


class PatchEmbedding(nn.Module):



  def __init__(self,config):
    super().__init__()

    self.config = config

    self.n_patches = config.n_patch
    self.patch_dim = config.im_channels* config.patch_size ** 2


    #patch embedding
    self.patch_embd = nn.Sequential(
        nn.LayerNorm(self.patch_dim),
        nn.Linear(self.patch_dim,config.n_embd),
        nn.LayerNorm(config.n_embd)
    )

    #cls tokens
    self.cls_token = nn.Parameter(torch.randn((config.n_embd,),device=device))

    #possitional embedding
    self.pos_embd = nn.Embedding(self.n_patches+1,config.n_embd)    # +1 for cls token





  def forward(self,x):

    B,C,H,W = x.shape

    # B,C,H,W -> B, n_patches , patch_dim    # patch_dim = C* patch_size*patch_size

    patch_size = self.config.patch_size

    patches = F.unfold(x, patch_size, stride = patch_size).transpose(-1,-2)

    #patch embedding
    patch_embd = self.patch_embd(patches)        # B, n_patches , n_embd

    #class token
    class_tok = self.cls_token.expand(B,1,-1)     # B , 1 , n_embd


    patch_embd = torch.cat((class_tok,patch_embd),dim =1 )    # B, n_patches +1  , n_embd

    #positional embedding
    pos_embd = self.pos_embd(torch.arange(0,self.n_patches+1,device=device))     # B, n_patches +1  , n_embd

    out = patch_embd + pos_embd

    return out





#_______________________________________________________________________________



class Block(nn.Module):


  def __init__(self,config):
    super().__init__()



    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = Attention(config.n_embd,config.n_head)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config.n_embd)


  def forward(self,x):

    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))

    return x


#_______________________________________________________________________________


class MLP(nn.Module):


  def __init__(self,n_embd):
    super().__init__()


    self.layer = nn.Linear(n_embd,4*n_embd)
    self.gelu = nn.GELU()
    self.proj = nn.Linear(4*n_embd,n_embd)
    self.dropout = nn.Dropout(0.1)


  def forward(self,x):


    x = self.gelu(self.layer(x))
    x = self.proj(x)
    x = self.dropout(x)

    return x


#_______________________________________________________________________________


class Attention(nn.Module):


  def __init__(self,n_embd,n_head) :
    super().__init__()

    self.nh = n_head

    self.w = nn.Linear(n_embd,3*n_embd)    # 3 * n_head * head_size
    self.proj = nn.Linear(n_embd,n_embd)


  def forward(self,x):

    B,T,C = x.shape

    wei = self.w(x)        # B,T, 3* C

    k,q,v = torch.chunk(wei,3, dim = -1)      # each B,T,C

    head_size = C//self.nh

    key   = k.view(B, T, self.nh, head_size).transpose(1, 2)    # B, n_head, T, head_size
    query = q.view(B, T, self.nh, head_size).transpose(1, 2)    # ""
    value = v.view(B, T, self.nh, head_size).transpose(1, 2)


    # weight = ( query @ key.transpose(-1,-2) )  * (head_size ** -0.5)    #B,nh,T,T
    # weight = F.softmax(weight,dim = -1)

    # out = weight @ value      #B,nh,T,n_head


    #Flash Attention
    out = F.scaled_dot_product_attention(query,key,value)



    out = out.transpose(1,2).view(B,T,C)    #B,T,nh,n_head

    out = self.proj(out)

    return out


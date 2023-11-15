import torch
from torch import nn, einsum
import torch.nn.functional as F
import math

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# ViT
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# ViT & CrossViT
# PreNorm class for layer normalization before passing the input to another function (fn)
class PreNorm(nn.Module):    
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# ViT & CrossViT
# FeedForward class for a feed forward neural network consisting of 2 linear layers, 
# where the first has a GELU activation followed by dropout, then the next linear layer
# followed again by dropout
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# ViT & CrossViT
# Attention class for multi-head self-attention mechanism with softmax and dropout
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.1):
        """

        :param dim: dimensionality of the input features, size of token embeddings
        :param heads: number of heads
        :param dim_head: dimension of a single head, length of query/key/value
        :param dropout: probability of keep
        """
        super().__init__()
        # set heads and scale (=sqrt(dim_head))
        # TODO
        self.dim = dim  # shape[-1] of input tensor
        self.heads = heads  # number of heads
        self.dim_head = dim_head
        self.scale = math.sqrt(dim_head)
        # we need softmax layer and dropout
        # TODO
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)
        # as well as the q linear layer
        # TODO
        self.w_q = nn.Linear(in_features=dim, out_features=self.heads * self.dim_head, bias=False)
        # and the k/v linear layer (can be realized as one single linear layer or as two individual ones)
        self.w_k = nn.Linear(in_features=dim, out_features=self.heads * self.dim_head, bias=False)
        self.w_v = nn.Linear(in_features=dim, out_features=self.heads * self.dim_head, bias=False)
        # TODO
        # and the output linear layer followed by dropout
        self.w_o = nn.Linear(in_features=self.heads * dim_head, out_features=dim)

    def forward(self, x, context=None, kv_include_self=False):
        # now compute the attention/cross-attention
        # in cross attention: x = class token, context = token embeddings
        # don't forget the dropout after the attention
        # and before the multiplication w. 'v'
        # the output should be in the shape 'b n (h d)'
        # b, n, _, h = *x.shape, self.heads
        # b for batch size, n is the number of tokens and d for dim, h for number of heads
        if context is None:
            context = x

        if kv_include_self:
            # cross attention requires CLS token includes itself as key / value
            context = torch.cat((x, context), dim=1)

        # TODO: attention
        # 1. dot product with weight matrices
        Q, K, V = self.w_q(x), self.w_k(context), self.w_v(context)

        # 2. split tensor by number of heads
        q, k, v, = self.split(Q), self.split(K), self.split(V)
        # 3. do scale dot product to compute similarity/attention as h_i
        # TODO: masking?
        h_i = self.scale_dot_product(q=q, k=k, v=v, mask=None)
        # 4. concat h_i
        H = self.concat_h(h_i=h_i)
        # 5. pass to linear layer
        M = self.w_o(H)
        # 6. add dropout if is needed
        out = self.dropout(M)
        return out

    def split(self, tensor):
        """
        split the long tensor into #heads sub-tensor
        :param tensor: [b, n, (h*d)]
        :return: [b, h, n, d]
        """
        b, n, d_model = tensor.size()
        d_tensor = d_model // self.heads
        tensor = tensor.view(b, n, self.heads, d_tensor).transpose(1, 2)

        return tensor

    def scale_dot_product(self, q, k, v, mask=None):
        """

        :param q: query, what a token is looking for, global information
        :param k: key, description of a query, local info
        :param v: value
        :param mask: if we use masking operation to the scaled value
        :return: attention score
        """
        b, h, n, d = k.size()
        k_T = k.transpose(2, 3)  # swap n and d
        scaled = q @ k_T / self.scale
        # TODO: if add mask
        if mask is not None:
            scaled = scaled.masked_fill(mask == 0, -1e9)
        score = self.dropout(self.softmax(scaled)) @ v
        return score

    def concat_h(self, h_i):
        """

        :param h_i: attention from single head [b,h,n,d]
        :return: concatenation of h_i as H [b, n, (h*d)]
        """
        b, h, n, d = h_i.size()
        H = h_i.transpose(1, 2).contiguous().view(b, n, h * d)
        return H


# ViT & CrossViT
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# CrossViT
# projecting CLS tokens, in the case that small and large patch tokens have different dimensions
class ProjectInOut(nn.Module):
    """
    Adapter class that embeds a callable (layer) and handles mismatching dimensions
    """
    def __init__(self, dim_outer, dim_inner, fn):
        """
        Args:
            dim_outer (int): Input (and output) dimension.
            dim_inner (int): Intermediate dimension (expected by fn).
            fn (callable): A callable object (like a layer).
        """
        super().__init__()
        self.fn = fn
        need_projection = dim_outer != dim_inner
        # TODO: I mistakenly defined the in/output shape of Linear layer
        self.project_in = nn.Linear(dim_outer, dim_inner) if need_projection else nn.Identity()     # f^l()
        self.project_out = nn.Linear(dim_inner, dim_outer) if need_projection else nn.Identity()    # g^l()

    def forward(self, x, *args, **kwargs):
        """
        Args:
            *args, **kwargs: to be passed on to fn

        Notes:
            - after calling fn, the tensor has to be projected back into it's original shape
            - fn(W_in) * W_out
        """
        # TODO
        x = self.project_in(x)  # f
        x = self.fn(x, *args, **kwargs)  # cross attention
        x = self.project_out(x)     # g
        return x


# CrossViT
# cross attention transformer
class CrossTransformer(nn.Module):
    # This is a special transformer block
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        # TODO: create # depth encoders using ProjectInOut
        # Note: no positional FFN here
        for i in range(depth):
            # cross attention for large branch
            small_attention_layer = Attention(dim=sm_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            crs_lg = ProjectInOut(dim_outer=lg_dim, dim_inner=sm_dim, fn=small_attention_layer)

            # cross attention for small branch
            large_attention_layer = Attention(dim=lg_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            crs_sm = ProjectInOut(dim_outer=sm_dim, dim_inner=lg_dim, fn=large_attention_layer)

            self.layers.append(nn.ModuleList([crs_lg, crs_sm]))

    def forward(self, sm_tokens, lg_tokens):
        # separate the first token (possibly a class token)
        # from the rest of the tokens (likely patch tokens) for two sets of tokens
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))
        # sm_cls: 16 1 64 sm_patch: 16 16 64
        # lg_cls: 16 1 128 lg_patch 16 4 128
        # Forward pass through the layers,
        # TODO
        for crs_lg, crs_sm in self.layers:
            # cross attention of large branch
            sm2lg = crs_lg(lg_cls, context=sm_patch_tokens)
            lg_cls = sm2lg + lg_cls  # skip links

            # cross attention of small branch
            lg2sm = crs_sm(sm_cls, context=lg_patch_tokens)
            sm_cls = lg2sm + sm_cls  # skip links

        # finally concat sm/lg cls tokens with patch tokens
        # TODO
        sm_tokens = torch.cat([sm_cls, sm_patch_tokens], dim=1)
        lg_tokens = torch.cat([lg_cls, lg_patch_tokens], dim=1)
        return sm_tokens, lg_tokens


# CrossViT
# multi-scale encoder
class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # TODO
                # 2 transformer branches, one for small, one for large patchs
                Transformer(dim=sm_dim, dropout=dropout, **sm_enc_params),
                Transformer(dim=lg_dim, dropout=dropout, **lg_enc_params),
                # + 1 cross transformer block
                CrossTransformer(sm_dim=sm_dim, lg_dim=lg_dim, depth=cross_attn_depth,
                                 heads=cross_attn_heads, dim_head=cross_attn_dim_head, dropout=dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        # forward through the transformer encoders and cross attention block
        # TODO
        for sm_enc, lg_enc, cross_attn in self.layers:
            sm_tokens = sm_enc(sm_tokens)
            lg_tokens = lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attn(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens

# CrossViT (could actually also be used in ViT)
# helper function that makes the embedding from patches
# have a look at the image embedding in ViT
class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2     # (p1 p2 c)
        patch_height, patch_width = pair(patch_size)
        # create layer that re-arranges the image patches
        # and embeds them with layer norm + linear projection + layer norm
        self.to_patch_embedding = nn.Sequential(
            # TODO
            # similar to the patch embedding in ViT
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        # create/initialize #dim-dimensional positional embedding (will be learned)
        # TODO
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))     # transfer a tensor to be learnable
        # create #dim cls tokens (for each patch embedding)
        # TODO
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # create dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        # TODO same as in ViT
        # patch embedding
        x = self.to_patch_embedding(img)    # b n dim
        b, n, d = x.size()  # batch size, number of patches, dim
        # concat class tokens
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        # and add positional embedding
        x += self.pos_embedding[:, :(n + 1)]
        # apply dropout
        return self.dropout(x)


# normal ViT
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # initialize patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))   # for each class, we use #dim tokens to represent it
        self.dropout = nn.Dropout(emb_dropout)

        # create transformer blocks
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # create mlp head (layer norm + linear layer)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # patch embedding
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # concat class token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        # add positional embedding
        x += self.pos_embedding[:, :(n + 1)]
        # apply dropout
        x = self.dropout(x)

        # forward through the transformer blocks
        x = self.transformer(x)

        # decide if x is the mean of the embedding 
        # or the class token
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        # transfer via an identity layer the cls tokens or the mean
        # to a latent space, which can then be used as input
        # to the mlp head
        x = self.to_latent(x)
        return self.mlp_head(x)


# CrossViT
class CrossViT(nn.Module):
    def __init__(
            self,
            *,
            image_size, # input image size
            num_classes, # number of classes to predict
            sm_dim, # dimension of small patches
            lg_dim, # dimension of large patches
            sm_patch_size=12, # size of small patches (how fine will the image be divided into patches)
            sm_enc_depth=1, # number of transformer encoder blocks for small patches
            sm_enc_heads=8, # number of heads for small patches
            sm_enc_mlp_dim=2048, # dimension of the feed forward network for small patches
            sm_enc_dim_head=64, # dimension of the head for small patches
            lg_patch_size=16,
            lg_enc_depth=4,
            lg_enc_heads=8,
            lg_enc_mlp_dim=2048,
            lg_enc_dim_head=64,
            cross_attn_depth=2,
            cross_attn_heads=8,
            cross_attn_dim_head=64,
            depth=3,
            dropout=0.1,
            emb_dropout=0.1
    ):
        super().__init__()
        # create ImageEmbedder for small and large patches
        # TODO
        self.small_img_embedder = ImageEmbedder(dim=sm_dim, image_size=image_size,
                                                patch_size=sm_patch_size, dropout=emb_dropout)
        self.large_img_embedder = ImageEmbedder(dim=lg_dim, image_size=image_size,
                                                patch_size=lg_patch_size, dropout=emb_dropout)
        # create MultiScaleEncoder
        self.multi_scale_encoder = MultiScaleEncoder(
            depth=depth,
            sm_dim=sm_dim,
            lg_dim=lg_dim,
            cross_attn_heads=cross_attn_heads,
            cross_attn_dim_head=cross_attn_dim_head,
            cross_attn_depth=cross_attn_depth,
            sm_enc_params=dict(
                depth=sm_enc_depth,
                heads=sm_enc_heads,
                mlp_dim=sm_enc_mlp_dim,
                dim_head=sm_enc_dim_head
            ),
            lg_enc_params=dict(
                depth=lg_enc_depth,
                heads=lg_enc_heads,
                mlp_dim=lg_enc_mlp_dim,
                dim_head=lg_enc_dim_head
            ),
            dropout=dropout
        )

        # create mlp heads for small and large patches
        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))

    def forward(self, img):
        # apply image embedders
        # TODO
        sm_tokens = self.small_img_embedder(img)
        lg_tokens = self.large_img_embedder(img)
        # and the multi-scale encoder
        # TODO
        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        # call the mlp heads w. the class tokens
        # TODO
        sm_logits = self.sm_mlp_head(sm_tokens[:, 0])
        lg_logits = self.lg_mlp_head(lg_tokens[:, 0])
        return sm_logits + lg_logits


if __name__ == "__main__":
    x = torch.randn(16, 3, 32, 32)
    vit = ViT(image_size = 32, patch_size = 8, num_classes = 10, dim = 64, depth = 2, heads = 8, mlp_dim = 128, dropout = 0.1, emb_dropout = 0.1)
    cvit = CrossViT(image_size = 32, num_classes = 10, sm_dim = 64, lg_dim = 128, sm_patch_size = 8,
                    sm_enc_depth = 2, sm_enc_heads = 8, sm_enc_mlp_dim = 128, sm_enc_dim_head = 64,
                    lg_patch_size = 16, lg_enc_depth = 2, lg_enc_heads = 8, lg_enc_mlp_dim = 128,
                    lg_enc_dim_head = 64, cross_attn_depth = 2, cross_attn_heads = 8, cross_attn_dim_head = 64,
                    depth = 3, dropout = 0.1, emb_dropout = 0.1)
    print(vit(x).shape)
    print(cvit(x).shape)

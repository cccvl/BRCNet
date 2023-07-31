import math

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            'Network [%s] was created. Total number of parameters: %.1f million. '
            'To see the architecture, do print(network).'
            % (type(self).__name__, num_params / 1000000)
        )

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (
                classname.find('Conv') != -1 or classname.find('Linear') != -1
            ):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented'
                        % init_type
                    )
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

class FeedForward2D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeedForward2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=3, padding=2, dilation=2
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

def attention(query, key, value):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
        query.size(-1)
    )
    p_attn = F.softmax(scores, dim=-1)
    p_val = torch.matmul(p_attn, value)
    return p_val, p_attn

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # d_k = c // len(self.patchsize)  #self.patchsize=[(19,19),(38,38)]
        output = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        attentions = []
        for (width, height), query, key, value in zip(
            self.patchsize,
            torch.chunk(_query, len(self.patchsize), dim=1),
            torch.chunk(_key, len(self.patchsize), dim=1),
            torch.chunk(_value, len(self.patchsize), dim=1),    #torch.chunk(tensor,chunk数，维度）
        ):
            out_w, out_h = w // width, h // height
            d_k = query.size(1)
            # 1) embedding and reshape
            query = query.view(b, d_k, out_h, height, out_w, width) #->shape=(b, d_k, out_h, height, out_w, width)
            query = (
                query.permute(0, 2, 4, 1, 3, 5)  #=transpos -> shape=(b, out_h, out_w, d_k, height, width)
                .contiguous()
                .view(b, out_h * out_w, d_k * height * width) #-> shape=(b, out_h * out_w, d_k * height * width)
            )
            key = key.view(b, d_k, out_h, height, out_w, width)
            key = (
                key.permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(b, out_h * out_w, d_k * height * width)
            )
            value = value.view(b, d_k, out_h, height, out_w, width)
            value = (
                value.permute(0, 2, 4, 1, 3, 5)
                .contiguous()
                .view(b, out_h * out_w, d_k * height * width)
            )

            y, _ = attention(query, key, value)

            # 3) "Concat" using a view and apply a final linear.
            y = y.view(b, out_h, out_w, d_k, height, width)
            y = y.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, d_k, h, w) #->to raw
            attentions.append(y)
            output.append(y)

        output = torch.cat(output, 1)
        self_attention = self.output_linear(output)

        return self_attention

class PatchTrans(BaseNetwork):
    def __init__(self, in_channel, in_size):
        super(PatchTrans, self).__init__()
        self.in_size = in_size
        #patchsize = [(in_size, in_size)]
        if in_size==76:
            patchsize = [
                (in_size, in_size),
                #(in_size // 2, in_size // 2),
                #(in_size // 4, in_size // 4)
            ]
        elif in_size==38:
            patchsize = [
                (in_size, in_size),
                #(in_size // 2, in_size // 2)
            ]
        else:
            patchsize = [
                (in_size, in_size)
            ]

        self.t = TransformerBlock(patchsize, in_channel=in_channel)

    def forward(self, enc_feat):
        output = self.t(enc_feat)
        return output

class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, patchsize, in_channel):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=in_channel)
        self.feed_forward = FeedForward2D(
            in_channel=in_channel, out_channel=in_channel
        )

    def forward(self, rgb):
        self_attention = self.attention(rgb)
        output = rgb + self_attention
        output = output + self.feed_forward(output)
        return output

if __name__ == "__main__":
    patch1=torch.randn(2, 728, 19, 19)
    patch2=torch.randn(2, 256, 38, 38)
    patch3=torch.randn(2, 128, 76, 76)
    bs, c, h, w = patch1.size()
    transformer=PatchTrans(c,h)
    a=transformer(patch1)
    print(a.size())




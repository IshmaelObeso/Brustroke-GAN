import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.callbacks import hook_outputs
from fastai.torch_core import requires_grad, children
from fastai.vision.gan import basic_generator, gan_critic

class Generator(nn.Module):
    
    def __init__(self, condition_dim, fc_dim, in_size, n_channels, n_extra_layers):
        super().__init__()
        self.fc = nn.Linear(condition_dim, fc_dim)
        self.generator = basic_generator(in_size=in_size, 
                                         n_channels=n_channels, 
                                         n_extra_layers=n_extra_layers, 
                                         noise_sz=fc_dim)
        
    def forward(self,x):
        x = self.fc(x)
        x = self.generator(x.view(x.size(0), x.size(1), 1, 1))
        return x
    
class FeatureLoss(nn.Module):
    def __init__(self, feature_extractor, layer_ids, layer_wgts):
        super(FeatureLoss, self).__init__()
        
        # using L1 as base for computing the loss
        self.base_loss = F.l1_loss 
        self.feature_extractor = feature_extractor
        self.loss_features = [self.feature_extractor[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts

    def make_features(self, x, clone=False):
        self.feature_extractor(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):

        input_features = self.make_features(input)
        target_features = self.make_features(target, clone=True)
        
        self.feat_losses = []
        
        self.feat_losses += [self.base_loss(f_in, f_out) * w
                             for f_in, f_out, w in zip(input_features, target_features, self.wgts)]
        

        self.feat_losses += [self.base_loss(gram_matrix(f_in), gram_matrix(f_out)) * w**2 * 5e3
                             for f_in, f_out, w in zip(input_features, target_features, self.wgts)]
    
        self.feat_losses += [self.base_loss(input, target)]
                
        return sum(self.feat_losses)
    
    def __del__(self):
        self.hooks.remove()
        
        
def gram_matrix(x):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features @ features_t / (ch * h * w)
    return gram

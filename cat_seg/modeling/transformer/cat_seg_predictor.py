# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified by Jian Ding from: https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
# Modified by Heeseong Shin from: https://github.com/dingjiansw101/ZegFormer/blob/main/mask_former/mask_former_model.py
import fvcore.nn.weight_init as weight_init
import torch

from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .model import Aggregator
from cat_seg.third_party import clip
from cat_seg.third_party import imagenet_templates

import numpy as np
class CATSegPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        train_class_json: str,
        test_class_json: str,
        clip_pretrained: str,
        prompt_ensemble_type: str,
        text_guidance_dim: int,
        text_guidance_proj_dim: int,
        appearance_guidance_dim: int,
        appearance_guidance_proj_dim: int,
        prompt_depth: int,
        prompt_length: int,
        decoder_dims: list,
        decoder_guidance_dims: list,
        decoder_guidance_proj_dims: list,
        num_heads: int,
        num_layers: tuple,
        hidden_dims: tuple,
        pooling_sizes: tuple,
        feature_resolution: tuple,
        window_sizes: tuple,
        attention_type: str,
        device: str = None,
        aggregation: str = 'serial',
        context_adapter: str = 'none',
        lsk_dilation: int = 3,
        lsk_scale: float = 0.01,
        epl_experts: int = 4,
        fusion_scale: float = 0.01,
        cad_stages: tuple = (),
        cad_scale: float = 0.01,
        cad_pooling: str = 'mean',
        epl_routing: str = 'learned',
        cad_scope: str = 'shared',
        cad_mode: str = 'residual',
    ):
        """
        Args:
            
        """
        super().__init__()
        
        import json
        # use class_texts in train_forward, and test_class_texts in test_forward
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
  
        self.tokenizer = None
        if clip_pretrained == "ViT-G" or clip_pretrained == "ViT-H":
            import open_clip
            # for OpenCLIP models
            name, pretrain = ('ViT-H-14', 'laion2b_s32b_b79k') if clip_pretrained == 'ViT-H' else ('ViT-bigG-14', 'laion2b_s39b_b160k')
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                name, 
                pretrained=pretrain, 
                device=device, 
                force_image_size=336,)
        
            self.tokenizer = open_clip.get_tokenizer(name)
        else:
            # for OpenAI models
            clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False, prompt_depth=prompt_depth, prompt_length=prompt_length)
    
        self.prompt_ensemble_type = prompt_ensemble_type        

        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = ['A photo of a {} in the scene',]
        else:
            raise NotImplementedError
        
        self.prompt_templates = prompt_templates

        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess
        
        transformer = Aggregator(
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=num_heads, 
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            feature_resolution=feature_resolution,
            window_size=window_sizes,
            attention_type=attention_type,
            prompt_channel=len(prompt_templates),
            aggregation=aggregation,
            context_adapter=context_adapter,
            lsk_dilation=lsk_dilation,
            lsk_scale=lsk_scale,
            epl_experts=epl_experts,
            fusion_scale=fusion_scale,
            cad_stages=cad_stages,
            cad_scale=cad_scale,
            cad_pooling=cad_pooling,
            epl_routing=epl_routing,
            cad_scope=cad_scope,
            cad_mode=cad_mode,
            )
        self.transformer = transformer
        
        self._token_cache = {}
        self._evaluation_cache = {}

    @classmethod
    def from_config(cls, cfg):#, in_channels, mask_classification):
        ret = {}

        ret["train_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON
        ret["test_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
        ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        ret["device"] = cfg.MODEL.DEVICE
        ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE

        # Aggregator parameters:
        ret["text_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM
        ret["text_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM
        ret["appearance_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM
        ret["appearance_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM

        ret["decoder_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS
        ret["decoder_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS
        ret["decoder_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS

        ret["prompt_depth"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH
        ret["prompt_length"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH

        ret["num_layers"] = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
        ret["num_heads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
        ret["hidden_dims"] = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS
        ret["pooling_sizes"] = cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES
        ret["feature_resolution"] = cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION
        ret["window_sizes"] = cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES
        ret["attention_type"] = cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE
        ret['aggregation'] = cfg.MODEL.OVRS.AGGREGATION
        ret['context_adapter'] = cfg.MODEL.OVRS.CONTEXT_ADAPTER
        ret['lsk_dilation'] = cfg.MODEL.OVRS.LSK_DILATION
        ret['lsk_scale'] = cfg.MODEL.OVRS.LSK_SCALE
        ret['epl_experts'] = cfg.MODEL.OVRS.EPL_EXPERTS
        ret['fusion_scale'] = cfg.MODEL.OVRS.FUSION_SCALE
        ret['cad_stages'] = tuple(cfg.MODEL.OVRS.CAD_STAGES)
        ret['cad_scale'] = cfg.MODEL.OVRS.CAD_SCALE
        ret['cad_pooling'] = cfg.MODEL.OVRS.CAD_POOLING
        ret['epl_routing'] = cfg.MODEL.OVRS.EPL_ROUTING
        ret['cad_scope'] = cfg.MODEL.OVRS.CAD_SCOPE
        ret['cad_mode'] = cfg.MODEL.OVRS.CAD_MODE

        return ret

    def forward(self, x, vis_guidance, prompt=None, gt_cls=None):
        vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]
        text = self.class_texts if self.training else self.test_class_texts
        text = [text[c] for c in gt_cls] if gt_cls is not None else text
        text = self.get_text_embeds(text, self.prompt_templates, self.clip_model, prompt)
        
        text = text.repeat(x.shape[0], 1, 1, 1)
        out = self.transformer(x, text, vis)
        return out

    @torch.no_grad()
    def class_embeddings(self, classnames, templates, clip_model):
        device = next(clip_model.parameters()).device
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]  # format with class
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).to(device)
            else: 
                texts = clip.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        return zeroshot_weights
    
    def train(self, mode=True):
        if self.training != mode:
            self._evaluation_cache.clear()
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self._evaluation_cache.clear()
        return super()._load_from_state_dict(*args, **kwargs)

    def get_text_embeds(self, classnames, templates, clip_model, prompt=None):
        device = next(clip_model.parameters()).device
        key = (tuple(classnames), tuple(templates))
        cache_key = (key, str(device), next(clip_model.parameters()).dtype)
        if not self.training and prompt is None and cache_key in self._evaluation_cache:
            return self._evaluation_cache[cache_key]
        if key not in self._token_cache:
            texts = [template.format(name.split(', ')[0])
                     for name in classnames for template in templates]
            tokenizer = self.tokenizer if self.tokenizer is not None else clip.tokenize
            self._token_cache[key] = tokenizer(texts).cpu()
        tokens = self._token_cache[key].to(device)
        if self.tokenizer is not None:
            if prompt is not None:
                raise ValueError('Prompt arguments are supported only for the bundled OpenAI CLIP')
            embeddings = clip_model.encode_text(tokens)
        else:
            embeddings = clip_model.encode_text(tokens, prompt)
        embeddings = F.normalize(embeddings, dim=-1)
        embeddings = embeddings.reshape(len(classnames), len(templates), -1)
        if not self.training and prompt is None:
            self._evaluation_cache[cache_key] = embeddings.detach()
        return embeddings

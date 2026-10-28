"""Extract frozen CLIP and RSIB features for the fixed DLRSD pilot."""
import argparse, hashlib, importlib.util, importlib.metadata, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import open_clip
from torchvision.transforms import functional as TF

CLIP_MEAN=(0.48145466,0.4578275,0.40821073)
CLIP_STD=(0.26862954,0.26130258,0.27577711)
DINO_MEAN=(0.485,0.456,0.406)
DINO_STD=(0.229,0.224,0.225)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def deduplicate_descriptions(rows):
    rows = [list(dict.fromkeys(row)) for row in rows]
    if any(not row or any(not text.strip() for text in row) for row in rows):
        raise ValueError('Each class needs nonempty descriptions')
    return rows


def main():
    p=argparse.ArgumentParser()
    for x in ('manifest','protocol','clip','rsib','rsib_source','output'): p.add_argument('--'+x.replace('_','-'),required=True)
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    manifest=Path(a.manifest); info=json.loads(manifest.read_text()); protocol=json.loads(Path(a.protocol).read_text())
    if sha(manifest)!=protocol['manifest_sha256']: raise ValueError('manifest identity mismatch')
    classes=json.loads((manifest.parent/info['train_classes']).read_text())
    if classes!=protocol['class_names']: raise ValueError('class order mismatch')
    full_classes=json.loads((manifest.parent/info['test_classes']).read_text())
    if classes != [full_classes[i] for i in info['seen_class_ids']]:
        raise ValueError('seen/original class mapping mismatch')
    hashes_before=[sha(a.clip),sha(a.rsib)]
    if hashes_before != [protocol['encoder_weights_sha256']['clip'], protocol['encoder_weights_sha256']['rsib']]:
        raise ValueError('encoder weight identity differs from protocol')
    prompts=deduplicate_descriptions([[template.format(**{'class': name}) for template in protocol['templates']] for name in classes])
    spec=importlib.util.spec_from_file_location('rsib_vit',a.rsib_source); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    rsib=mod.vit_base(patch_size=8,num_classes=0)
    with torch.serialization.safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype('float64')), argparse.Namespace]):
        ck=torch.load(a.rsib,map_location='cpu',weights_only=True)
    sd=ck['teacher'] if 'teacher' in ck else ck
    sd={k.removeprefix('module.').removeprefix('backbone.'):v for k,v in sd.items()}
    result=rsib.load_state_dict(sd,strict=False)
    if result.missing_keys or any(not k.startswith('head.') for k in result.unexpected_keys):
        raise RuntimeError('RSIB backbone load mismatch: '+str(result))
    clip=open_clip.load_openai_model(a.clip,device='cuda',precision='fp32',jit=False).eval()
    rsib=rsib.cuda().eval()
    for m in (clip,rsib): m.requires_grad_(False)
    states_before=[state_hash(clip), state_hash(rsib)]
    tokenizer=open_clip.get_tokenizer('ViT-B-16')
    with torch.inference_mode():
        encoded=clip.encode_text(tokenizer([x for row in prompts for x in row]).cuda()).cpu()
        text=torch.zeros(len(classes), max(map(len,prompts)), encoded.shape[-1])
        valid=torch.zeros(text.shape[:2], dtype=torch.bool)
        offset=0
        for i, row in enumerate(prompts):
            text[i,:len(row)]=encoded[offset:offset+len(row)]
            valid[i,:len(row)]=True
            offset+=len(row)
        for split in ('train','val'):
            vs=[]; ss=[]; ys=[]; records=[]
            for index in protocol['indices'][split]:
                record=info[split][index]; ip=Path(record['file_name']); mp=manifest.parent/record['sem_seg_file_name']
                image=Image.open(ip).convert('RGB').resize((224,224),Image.Resampling.BICUBIC)
                image=TF.to_tensor(image).unsqueeze(0).cuda()
                x=TF.normalize(image,CLIP_MEAN,CLIP_STD); visual=clip.visual
                z=visual.conv1(x).reshape(1,visual.conv1.out_channels,-1).permute(0,2,1)
                z=torch.cat([visual.class_embedding[None,None,:].expand(1,1,-1),z],dim=1)+visual.positional_embedding
                z=visual.ln_pre(visual.patch_dropout(z)).permute(1,0,2); z=visual.transformer(z).permute(1,0,2)
                projected=visual.ln_post(z) @ visual.proj
                torch.testing.assert_close(projected[:,0],visual(x),rtol=1e-5,atol=1e-5)
                vf=projected[:,1:].transpose(1,2).reshape(1,512,14,14)
                df=rsib.get_intermediate_layers(TF.normalize(image,DINO_MEAN,DINO_STD),n=1)[0][:,1:].transpose(1,2).reshape(1,768,28,28)
                mask=np.asarray(Image.open(mp).resize((224,224),Image.Resampling.NEAREST),dtype=np.int64).copy()
                if split=='val':
                    if not np.isin(mask,list(range(len(full_classes)))+[255]).all():
                        raise ValueError('invalid original validation labels '+str(mp))
                    mapped=np.full(mask.shape,255,dtype=np.int64)
                    for i,old in enumerate(info['seen_class_ids']): mapped[mask==old]=i
                    mask=mapped
                if not np.isin(mask,list(range(len(classes)))+[255]).all(): raise ValueError('invalid labels '+str(mp))
                vs.append(vf.cpu()); ss.append(df.cpu()); ys.append(torch.from_numpy(mask)); records.append({'index':index,'id':ip.stem,'split':split,'image':str(ip),'mask':str(mp),'image_sha256':sha(ip),'mask_sha256':sha(mp)})
                print(split,index,ip.stem,flush=True)
            metadata={'records':records,'manifest_sha256':sha(manifest),'classes':classes,'class_original_ids':info['seen_class_ids'],'descriptions':prompts,'clip_weight_sha256':hashes_before[0],'rsib_weight_sha256':hashes_before[1],'rsib_source_sha256':sha(a.rsib_source),'clip':'OpenAI ViT-B/16 final token ln_post and visual.proj; patch extrapolation','structure':'RSIB ViT-B/8 last normalized patch tokens','preprocess':'same 224x224 bicubic RGB resize; CLIP and ImageNet normalization; masks nearest','shapes':'visual 14x14, RSIB 28x28, label 224x224','ignore':255,'val_mapping':'original IDs to seen IDs, unseen ignored'}
            torch.save({'visual':torch.cat(vs),'text':text,'valid':valid,'structures':[torch.cat(ss)],'labels':torch.stack(ys),'metadata':metadata},out/(split+'.pt'))
            (out/(split+'_metadata.json')).write_text(json.dumps(metadata,indent=2))
    if [sha(a.clip),sha(a.rsib)]!=hashes_before: raise RuntimeError('weights changed')
    states_after=[state_hash(clip), state_hash(rsib)]
    if states_after != states_before: raise RuntimeError('encoder state changed')
    audit={'encoder_state_hashes_before':states_before, 'encoder_state_hashes_after':states_after,
           'trainable_encoder_parameters':sum(p.numel() for m in (clip,rsib) for p in m.parameters() if p.requires_grad),
           'rsib_missing_keys':result.missing_keys, 'rsib_unexpected_keys':result.unexpected_keys,
           'clip_normalization':[CLIP_MEAN,CLIP_STD], 'rsib_normalization':[DINO_MEAN,DINO_STD],
           'extractor_sha256':sha(__file__), 'open_clip_version':importlib.metadata.version('open-clip-torch')}
    (out/'encoder_audit.json').write_text(json.dumps(audit,indent=2))
    print('encoder parameters frozen:',all(not x.requires_grad for m in (clip,rsib) for x in m.parameters()),'load unexpected RSIB keys:',result.unexpected_keys,flush=True)
if __name__=='__main__': main()

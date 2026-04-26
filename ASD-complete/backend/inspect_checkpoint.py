import torch

checkpoint = torch.load('vit_asd_best.pth', weights_only=False, map_location=torch.device('cpu'))
print('Total parameters:', len(checkpoint))
print('\nSample tensor shapes:')
sample_keys = ['cls_token', 'pos_embed', 'patch_embed.proj.weight', 'head.0.weight', 'head.3.weight']
for key in sample_keys:
    if key in checkpoint:
        print(f'{key}: {checkpoint[key].shape}')
    else:
        print(f'{key}: NOT FOUND')

print('\n--- Complete Summary ---')
print(f'Checkpoint type: {type(checkpoint).__name__}')
print(f'Is state dict: {isinstance(checkpoint, dict)}')
print(f'Number of parameters: {len(checkpoint)}')
print('\nThis is a PyTorch MODEL STATE DICT (direct parameter dictionary)')
print('Ready to be loaded into a model using model.load_state_dict()')

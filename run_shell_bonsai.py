import os

for lmbda in [0.003, 0.002, 0.001]:  # Optionally, you can try: 0.003, 0.002, 0.001, 0.0005
    for cuda, scene in enumerate(['akazukin']):
        sfm_cmd = f'CUDA_VISIBLE_DEVICES={0} python convert.py -s ../workspace/data/bonsai/{scene}'
        train_cmd = f'CUDA_VISIBLE_DEVICES={0} python train.py -s ../workspace/data/bonsai/{scene} --eval --lod 0 --voxel_size 0.001 --update_init_factor 4 --iterations 30_000 -m ../workspace/data/bonsai/{scene}/outputs/{lmbda} --lmbda {lmbda}'
        os.system(sfm_cmd)
        os.system(train_cmd)

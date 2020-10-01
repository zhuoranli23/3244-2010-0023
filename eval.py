import pytorch_fid.fid_score as fidsc

coloured_dir = './results/coloured'
original_dir = './results/original'
batch_size = 1
dims = 2048 # Refer to pytorch_fid README for other options
gpu_ids = [0]

paths = [coloured_dir, original_dir]

fid_value = fidsc.calculate_fid_given_paths(paths, batch_size, len(gpu_ids) > 0, dims)
print('FID: ', fid_value)
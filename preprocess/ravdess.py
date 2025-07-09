import os
import glob
import decord

data_path = './dataset/ravdess'

save_dir = f'../saved/data/voxceleb2/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# read
video_files = glob.glob(os.path.join(data_path,'*/*/*.mp4'))
print(f'Total videos: {len(video_files)}')

# write
out_file = os.path.join(save_dir, f'info_clean.csv')

count = 0
with open(out_file, 'w') as f:
    for video_file in video_files:
        count += 1
        f.write(f'{video_file} 0\n') # 0 for not causing error: dummy label

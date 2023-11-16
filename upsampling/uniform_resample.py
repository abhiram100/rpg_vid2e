import glob
import cv2
import numpy as np
import tqdm
from pathlib import Path

###################

DT_UNIFORM  = 0.001  # 1000 FPS
SAVE_FOLDER = '/data/storage/abhiram/dsec/dsec-det/v2e_dsec_merged_root'

####################

def valid_dir(folder):
    if folder.split('/')[-1] == 'upsampled':
        return True
    return False

def find_nearest_timestamps(q, timestamps):
    v = np.abs(timestamps-q)
    idx_min = np.argmin(v)
    if np.isclose(np.min(v), 0) or \
       idx_min == 0 or \
       idx_min == len(timestamps)-1:
        return idx_min, idx_min
    
    d_lower = q - timestamps[idx_min-1]
    d_upper = timestamps[idx_min+1] - q
    
    if d_lower < d_upper:
        return idx_min-1, idx_min
    else:
        return idx_min, idx_min+1
    

def process_dir(folder):
    timestamps = np.loadtxt(folder+'/timestamps.txt')   # In seconds
    uniform_timestamps = np.arange(timestamps[0], timestamps[-1], DT_UNIFORM)
    
    save_dir = SAVE_FOLDER + '/' + '/'.join(folder.split('/')[-3:-1]) + '/resampled'
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
        
    for c, t in enumerate(tqdm.tqdm(uniform_timestamps)):
        idx_down, idx_up = find_nearest_timestamps(t, timestamps)
        if idx_down == idx_up:
            im_between = cv2.imread(folder+'/imgs/%08d.png' % idx_down, cv2.IMREAD_COLOR)   # Images do not have color anyways
        else:
            im_down = cv2.imread(folder+'/imgs/%08d.png' % idx_down, cv2.IMREAD_COLOR)
            im_up = cv2.imread(folder+'/imgs/%08d.png' % idx_up, cv2.IMREAD_COLOR)
            
            im_down = np.array(im_down)
            im_up = np.array(im_up)
            t_down = timestamps[idx_down]
            t_up = timestamps[idx_up]
            
            im_between = im_down*(t-t_down)/(t_up-t_down) + im_up*(t_up-t)/(t_up-t_down)
        
        cv2.imwrite(save_dir+'/%08d.png' % c, im_between)
        
if __name__=='__main__':
    root = '/data/storage/abhiram/dsec/dsec-det/vid2e_dsec_merged_root/test'
    
    for folder in glob.glob(root+'/**', recursive=True):
        if valid_dir(folder):
            process_dir(folder)
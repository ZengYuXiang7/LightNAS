import numpy as np
import pickle
import json
import ast

file_paths = [
    './datasets/desktop-cpu-core-i7-7820x-fp32.pickle',
    './datasets/desktop-gpu-gtx-1080ti-fp32.pickle',
    './datasets/desktop-gpu-gtx-1080ti-large.pickle',
    './datasets/embedded-gpu-jetson-nono-fp16.pickle',
    './datasets/embedded-tpu-edgetpu-large.pickle',
    './datasets/embeeded-tpu-edgetpu-int8.pickle',
    './datasets/mobile-cpu-snapdragon-450-contex-a53-int8.pickle',
    './datasets/mobile-cpu-snapdragon-675-kryo-460-int8.pickle',
    './datasets/mobile-cpu-snapdragon-855-kryo-485-int8.pickle',
    './datasets/mobile-dsp-snapdragon-675-hexagon-685-int8.pickle',
    './datasets/mobile-dsp-snapdragon-855-hexagon-690-int8.pickle',
    './datasets/mobile-gpu-snapdragon-450-adreno-506-int8.pickle',
    './datasets/mobile-gpu-snapdragon-675-adren0-612-int8.pickle',
    './datasets/mobile-gpu-snapdragon-855-adren0-640-int8.pickle'
]


key_map = {}
idx = 0
first = True
all_df = np.zeros((len(file_paths), 15284, 6))
all_y = np.zeros((len(file_paths), 15284, 1))
for i, file_path in enumerate(file_paths):
    if first:
        idx = 0
        try:
            with open(file_path,'rb') as file:
                loaded_data = pickle.load(file)
                for key, value in loaded_data.items():
                    if key not in key_map:
                        all_df[i, idx] = key
                        key_map[key] = idx
                        all_y[i, idx] = value
                        idx += 1
        except Exception as e:
                print(f"Error loading1 {file_path}: {e}")
        first = False
    else:
        try:
            with open(file_path,'rb') as file:
                loaded_data = pickle.load(file)
                for key, value in loaded_data.items():
                    all_df[i, key_map[key]] = key 
                    all_y[i, key_map[key]] = value
        except Exception as e:
            print(f"Error loading2 {file_path}: {e}")
    print(file_path)
    
for i, file_path in enumerate(file_paths):
    x = all_df[i]
    y = all_y[i]
    data = np.concatenate((x, y), axis=1)
    output_file_path = f'./datasets/pickle/{file_path.split("/")[-1].replace(".pickle", "_data.pkl")}'

    with open(output_file_path,'wb') as output_file:
        pickle.dump(data,output_file)
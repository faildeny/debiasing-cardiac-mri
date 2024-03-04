import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, dataset_path, sample_weight_clipping=10):
        self.data = []
        self.dataset_path = dataset_path
        with open(dataset_path+'prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
        print("Loaded ", len(self.data), " samples.")
        self.sample_weight_clipping = sample_weight_clipping
        self.sample_weights = self.calculate_sample_weights()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        source = cv2.imread(self.dataset_path + source_filename)
        target = cv2.imread(self.dataset_path + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # # Resize source images.
        # source = cv2.resize(source, (size, size), interpolation=cv2.INTER_AREA)
        # target = cv2.resize(target, (size, size), interpolation=cv2.INTER_AREA)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source, filename=target_filename)
    
    def get_samples_list(self):
        return self.data
    
    def calculate_sample_weights(self):
        class_sizes = {}
        sample_weights = []
        for item in self.data:
            if item['prompt'] not in class_sizes:
                class_sizes[item['prompt']] = 0
            class_sizes[item['prompt']] += 1
        
        max_weight = 1 / (9*self.sample_weight_clipping) # 1% of the total number of samples

        class_sizes = dict(sorted(class_sizes.items(), key=lambda item: item[1], reverse=True))
        # for prompt, count in class_sizes.items():
            # print(prompt, count)
        
        # print number of classes
        # print("Number of classes: ", len(class_sizes))
        
        # Add weights for each sample.
        for item in self.data:
            sample_weight = 1.0 / class_sizes[item['prompt']]
            if sample_weight > max_weight:
                sample_weight = max_weight
                # print("Sample weight is too high, setting to max: ", sample_weight, " for prompt: ", item['prompt'])
            # else:
                # print("Sample weight: ", sample_weight, " for prompt: ", item['prompt'])
            # item['sample_weight'] = sample_weight
            sample_weights.append(sample_weight)

        return sample_weights

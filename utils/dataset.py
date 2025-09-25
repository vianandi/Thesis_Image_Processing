import os, random
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class LOLDataset(Dataset):
    def __init__(self, root_dir, split="train", split_ratio=(0.7,0.1,0.2), seed=42, transform=None):
        self.low_dir = os.path.join(root_dir, "low")
        self.high_dir = os.path.join(root_dir, "high")

        low_images = sorted(glob(os.path.join(self.low_dir, "*.png")))
        high_images = sorted(glob(os.path.join(self.high_dir, "*.png")))
        assert len(low_images) == len(high_images), "Low & High mismatch!"

        data = list(zip(low_images, high_images))
        random.seed(seed)
        random.shuffle(data)

        n = len(data)
        n_train = int(split_ratio[0]*n)
        n_val = int(split_ratio[1]*n)

        if split == "train":
            self.data = data[:n_train]
        elif split == "val":
            self.data = data[n_train:n_train+n_val]
        else:
            self.data = data[n_train+n_val:]

        self.transform = transform or T.Compose([
            T.Resize((128,128)),  # Ukuran lebih kecil untuk evaluasi cepat
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        low_path, high_path = self.data[idx]
        low = Image.open(low_path).convert("RGB")
        high = Image.open(high_path).convert("RGB")
        return self.transform(low), self.transform(high), os.path.basename(low_path)
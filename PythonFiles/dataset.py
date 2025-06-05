import torchvision.transforms as transforms
from config import img_size, img_channels
from torch.utils.data import Dataset
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self):
        horses_files = os.listdir("GAN_test/test/horses")+os.listdir("GAN_test/train/horses")
        humans_files = os.listdir("GAN_test/test/humans")+os.listdir("GAN_test/train/humans")

        self.image_files = horses_files+humans_files
        self.labels = [0]*len(horses_files)+[1]*len(humans_files)
        self.set = ['test']*128+['train']*500+['test']*128+['train']*527

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        trans = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            # transforms.Normalize([0.5 for _ in range(img_channels)],
                                 # [0.5 for _ in range(img_channels)]),
        ])
        filename = self.image_files[index]
        label = self.labels[index]
        convert = {0:'horses', 1:'humans'}
        image = Image.open(f"GAN_test/{self.set[index]}/{convert[label]}/{filename}").convert('RGB')
        image = trans(image) #Normalized
        return [image, label]

if __name__ == '__main__':
    pass

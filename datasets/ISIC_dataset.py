from torch.utils.data import Dataset
import os
import cv2
import torch


class ISICDataset(Dataset):

    def __init__(self, root, transform) -> None:
        super(ISICDataset, self).__init__()
        self.image_path = os.path.join(root, 'images')
        self.mask_path = os.path.join(root, 'masks')
        self.transform = transform
        self.imgs = os.listdir(self.image_path)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_path, self.imgs[index])
        mask_path = os.path.join(self.mask_path, self.imgs[index].split('.')[
                                 0]+'_segmentation.png')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return (img, mask, img_path)
    
class ToTensor(object):
    def __call__(self, image):
        # 从numpy转换为tensor
        image = torch.from_numpy(image).float()
        return image
        

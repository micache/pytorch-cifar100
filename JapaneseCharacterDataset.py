import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class JapaneseCharacterDataset(Dataset):
    def __init__(self, root_dir, dataset_type='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            dataset_type (string): 'train' or 'test' to indicate the dataset type.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, dataset_type)
        self.transform = transform
        self.image_files = []
        self.labels = []
        self.classes = []

        # Load image paths and labels
        for label in os.listdir(self.root_dir):
            label_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(label_dir):
                if label not in self.classes:
                    self.classes.append(label)
                class_index = self.classes.index(label)
                for file_name in os.listdir(label_dir):
                    if file_name.endswith('.png'):
                        self.image_files.append(os.path.join(label_dir, file_name))
                        self.labels.append(class_index)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

if __name__ == '__main__':

    # Define transformations for images
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create the dataset and DataLoader for training data
    train_dataset = JapaneseCharacterDataset(root_dir=os.getcwd() + '/raw/', dataset_type='train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
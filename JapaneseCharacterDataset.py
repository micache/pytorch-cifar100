import os, csv
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

class JapaneseCharacterDataset(Dataset):
    def __init__(self, root_dir, dataset_type='train', transform=None, max_seq_length=5, pad_idx=26):
        """
        Args:
            root_dir (string): Directory with all the images.
            dataset_type (string): 'train' or 'test' to indicate the dataset type.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.ov_root = root_dir
        self.root_dir = os.path.join(root_dir, dataset_type)
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.pad_idx = pad_idx
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
        
        seq_label = self.pad_label(self.parse_labels_file(self.ov_root + '/labels.txt').get(label))

        if self.transform:
            image = self.transform(image)
        
        return image, label, seq_label
    
    def pad_label(self, label):
        padded_label = [(ord(char) - 97) for char in label] + [self.pad_idx] * (self.max_seq_length - len(label))
        return np.array(padded_label)
    
    def parse_labels_file(self, file_path):
        id_to_sequence = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=' ')
            next(reader)  # Skip header
            for row in reader:
                class_id = int(row[0])
                cangjie_sequence = row[4]  # Assuming the Cangjie sequence is in the 4th column
                id_to_sequence[class_id] = cangjie_sequence
        
        return id_to_sequence

if __name__ == '__main__':

    # Define transformations for images
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Create the dataset and DataLoader for training data
    train_dataset = JapaneseCharacterDataset(root_dir=os.getcwd() + '/raw/', dataset_type='train', transform=transform)
    print (len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

    for image, label, seq in train_dataloader:
        print(seq.size(0))
        print(seq[0])
        true_seq = ''.join([chr(char + 97) if char != 26 else '!' for char in seq[0].cpu().numpy()])
        print(true_seq)
        break

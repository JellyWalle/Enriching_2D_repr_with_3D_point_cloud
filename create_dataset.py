import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class CIFAR100Subset(Dataset):
    def __init__(self, root='./data', train=True, transform=None, selected_classes=None):
        self.cifar100 = datasets.CIFAR100(root=root, train=train, download=True, transform=None)  # ← ВАЖНО!
        self.transform = transform

        if selected_classes is None:
            raise ValueError("selected_classes must be provided")

        self.class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}

        self.samples = []
        for i in range(len(self.cifar100)):
            img, old_label = self.cifar100[i]
            class_name = self.cifar100.classes[old_label]
            if class_name in self.class_to_idx:
                self.samples.append((img, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        if self.transform:
            img = self.transform(img)  # Теперь img — PIL Image → OK
        return img, label

    @property
    def classes(self):
        return list(self.class_to_idx.keys())
    

if __name__ == '__main__':
    from torchvision import transforms

    # Трансформации
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR100 stats
    ])

    selected_classes = [
        'computer_keyboard',
        'laptop',
        'motorcycle',
        'pickup_truck',
        'skyscraper',
        'television'
    ]

    # Загрузка
    train_dataset = CIFAR100Subset(root='./data', train=True, transform=transform, selected_classes=selected_classes)
    test_dataset = CIFAR100Subset(root='./data', train=False, transform=transform, selected_classes=selected_classes)

    print("Train size:", len(train_dataset))
    print("Test size:", len(test_dataset))
    print("Classes:", train_dataset.classes)
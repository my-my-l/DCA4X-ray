from PIL import Image
import os
import torchvision.transforms as T
from timm.data.auto_augment import auto_augment_transform, rand_augment_transform
import tllib.vision.datasets as datasets
from tllib.vision.transforms import ResizeImage
from torch.utils.data import DataLoader


class MedicalImages(ImageList):

    image_list = {
        "N": "NIH_CXR14",
        "C": "CheXpert",
        "M": "MIMIC_CXR",
        "O": "Open_i",
    }
    root_list = {
        "N": "NIH_CXR14",
        "C": "CheXpert",
        "M": "MIMIC_CXR",
        "O": "Open_i",
    }
     
    classes = ['0 - N', '1 - P']
    def __init__(self, root, task, class_index, split='train', download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        assert split in ['train', 'test', 'uncertain', 'train_without_uncertain', 'test_without_uncertain']
        data_list_file_path = os.path.join(root, self.root_list[task], 'image_list', '{}_{}_{}.txt'.format(self.image_list[task],str(class_index),split))
        super(MedicalImages, self).__init__(root, MedicalImages.get_classes(), data_list_file=data_list_file_path, **kwargs)


    @classmethod
    def get_classes(self):
        return self.classes

def get_dataset(root, source, target, train_source_transform, val_transform, train_target_transform=None, without_normal=False, sl=False, class_index=None):
    if train_target_transform is None:
        train_target_transform = train_source_transform
    train_source_dataset = MedicalImages(root, task=source[0], class_index=class_index, split='train',
                                                                download=True, transform=train_source_transform)
    train_target_dataset = MedicalImages(root, task=target[0], class_index=class_index, split='train',
                                                                download=True, transform=train_target_transform)
    source_val_dataset = source_test_dataset = MedicalImages(root, task=source[0], class_index=class_index, split='test',
                                                                                    download=True, transform=val_transform)
    target_val_dataset = target_test_dataset = MedicalImages(root, task=target[0], class_index=class_index, split='test',
                                                                                    download=True, transform=val_transform)
    class_names = datasets.MedicalImages.get_classes()
    num_classes = len(class_names)

    return train_source_dataset, train_target_dataset, target_val_dataset, target_test_dataset, source_val_dataset, source_test_dataset, num_classes, class_names


    

def get_train_transform(resizing='default', scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), random_horizontal_flip=True,
                        random_color_jitter=False, resize_size=224, norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225), auto_augment=None):
    transformed_img_size = 224
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224, scale=scale, ratio=ratio)
        ])
    elif resizing == 'cen.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224)
        ])
    elif resizing == 'ran.crop':
        transform = T.Compose([
            ResizeImage(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
        transformed_img_size = resize_size
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if auto_augment:
        aa_params = dict(
            translate_const=int(transformed_img_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in norm_mean]),
            interpolation=Image.BILINEAR
        )
        if auto_augment.startswith('rand'):
            transforms.append(rand_augment_transform(auto_augment, aa_params))
        else:
            transforms.append(auto_augment_transform(auto_augment, aa_params))
    elif random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])
    return T.Compose(transforms)


def get_val_transform(resizing='default', resize_size=224,
                      norm_mean=(0.485, 0.456, 0.406), norm_std=(0.229, 0.224, 0.225)):
    if resizing == 'default':
        transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res.':
        transform = ResizeImage(resize_size)
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std)
    ])

def send_to_device(tensor, device):
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)

class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)
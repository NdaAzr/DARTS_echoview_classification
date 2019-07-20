
import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
#from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from PIL import Image
import numpy 
import glob
import os
import os.path
from sklearn.model_selection import train_test_split

#find_classes and make_dataset are function from Pytorch source
#dir = 'D:/Neda/Echo_View_Classification/avi_images/'
dir = 'D:/Neda/NAS/DARTS/DARTS_echoview_classification/cnn/avi_images - Copy'

def find_classes(dir):   # Finds the class folders in a dataset, dir (string): Root directory path.
        
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        
        return classes, class_to_idx

def make_dataset(dir, class_to_idx): #map each image to correspond target(class id)
    images = []
    dir = os.path.expanduser(dir)
    
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)   
    return images

class CustomDataset_classification(Dataset):  
    
    def __init__(self, image_paths, targets, classes, images):  
        
        self.images = images
        self.image_paths = image_paths
        self.transforms = transforms.Compose([
            transforms.Resize(200), #224
            transforms.ToTensor(),
            transforms.Normalize([0.0698],[0.1523]),
            ])
        self.classes = classes
        self.targets = targets 
    
    def __getitem__(self, index):

        image = Image.open(self.image_paths[index])
        t_image = image.convert('L')
        t_image = self.transforms(t_image) 
        targets = self.targets[index]  
        targets = torch.tensor(targets, dtype=torch.long)
        #classes = self.classes[targets]
        return t_image, targets, self.classes[targets], self.image_paths[index]
    
    def __len__(self): 
        return len(self.image_paths)
    

find_classes(dir)

#folder_data = glob.glob("D:\\Neda\\Echo_View_Classification\\avi_images\\*\\*.png") 
folder_data = glob.glob("D:\\Neda/NAS\\DARTS\\DARTS_echoview_classification\\cnn\\avi_images - Copy\\*\\*.png") 

#classes, targets = find_classes('D:/Neda/Echo_View_Classification/avi_images/')  #avi_images _for_quick_run_test
classes, targets = find_classes('D:/Neda/NAS/DARTS/DARTS_echoview_classification/cnn/avi_images - Copy')  #avi_images _for_quick_run_test


#data_target = make_dataset('D:/Neda/Echo_View_Classification/avi_images/', targets)
data_target = make_dataset('D:/Neda/NAS/DARTS/DARTS_echoview_classification/cnn/avi_images - Copy', targets)

data_array, targets_array = zip(*data_target)

data= numpy.array(data_array)
targets_array = numpy.array(targets_array)

folder_data_train, folder_data_test, train_target, test_target = train_test_split(data, targets_array, test_size=0.20, random_state=42, shuffle = True, stratify = targets_array)
    
split_1 = int(0.75 * len(folder_data_train))
split_2 = int(0.25 * len(folder_data_train))

train_image_paths = folder_data_train[:split_1]
print(len(train_image_paths))
valid_image_paths = folder_data_train[split_1:]
print(len(valid_image_paths))


train_tragets = train_target[:split_1]
valid_targets = train_target[split_1:]


classes= ['A2CH', 'A3CH', 'A4CH_LV', 'A4CH_RV', 'A5CH', 'Apical_MV_LA_IAS',
 'OTHER', 'PLAX_TV', 'PLAX_full', 'PLAX_valves', 'PSAX_AV', 'PSAX_LV',
 'Subcostal_IVC', 'Subcostal_heart', 'Suprasternal']
  


 
      

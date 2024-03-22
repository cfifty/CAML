from torchvision.datasets import ImageFolder
import os

class FungiDataset(ImageFolder):

  def __init__(self, split: str = "train", transform=None):
    """class_column should be the attribute that we want to treat as classes. Either 'artist', 'genre', or 'style'"""
    assert split in ['train', 'val']
    path_to_datasets = '../image_datasets/'
    super().__init__(f'{path_to_datasets}fungi/{split}_images', transform=transform)

    self.target_to_index = {
      class_idx: []
      for class_idx in range(len(self.classes))
    }
    for sample_idx, target in enumerate(self.targets):
      self.target_to_index[target].append(sample_idx)
    self.all_targets = list(self.target_to_index.keys())


def rename_fungi_folders(split):

  read_path = f'../image_datasets/fungi/{split}_images'
  save_path = f'../image_datasets/fungi/{split}_images_v2'
  os.makedirs(save_path, exist_ok=True)
  class_paths = os.listdir(read_path)

  for i,class_name in enumerate(class_paths):
    if ' ' in class_name:
      os.system(f'mv {read_path}/"{class_name}" {read_path}/{i}')
    else:
      os.system(f'mv {read_path}/"{class_name}" {read_path}/{i}')



if __name__ == '__main__':
  # rename_fungi_folders('val')
  dataset = FungiDataset('train')
  for x in dataset:
    print(x)
    break

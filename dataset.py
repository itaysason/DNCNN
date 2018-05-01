import PIL.ImageOps
from PIL import Image
from torch.utils.data import Dataset


class DenoisedDataset(Dataset):

    def __init__(self, images_folder, noise_image_folder, patch_size=50, stride=20, transform=None, should_invert=True):
        self.images_folder = images_folder
        self.noise_images_folder = noise_image_folder
        self.transform = transform
        self.should_invert = should_invert
        self.num_images = len(images_folder.imgs)
        self.patch_size = patch_size
        self.stride = stride
        self.images = []
        self.patches_list = []
        self.noises_list = []
        # creating list of locations of all possible patches for each image
        for i in range(self.num_images):

            img0_tuple = self.images_folder.imgs[i]
            img1_tuple = self.noise_images_folder.imgs[i]
            source = Image.open(img0_tuple[0])
            target = Image.open(img1_tuple[0])
            source = source.convert("L")
            target = target.convert("L")

            if self.should_invert:
                source = PIL.ImageOps.invert(source)
                target = PIL.ImageOps.invert(target)

            if self.transform is not None:
                source = self.transform(source)
                target = self.transform(target)

            self.images.append(source)
            self.noises_list.append(target)
            source_height = source.shape[1]
            source_width = source.shape[2]
            possible_patches = source[:, 0:source_height - self.patch_size + 1:self.stride,
                               0:source_width - self.patch_size + 1:self.stride]

            for j in range(possible_patches.shape[1]):
                for k in range(possible_patches.shape[2]):
                    index_x = j * self.stride
                    index_y = k * self.stride
                    if index_x > 0:
                        index_x -= 1
                    if index_y > 0:
                        index_y -= 1
                    new_patch = (i, index_x, index_y)
                    self.patches_list.append(new_patch)

    def __getitem__(self, index):

        source_location = self.patches_list[index]
        source = self.images[source_location[0]]
        source = source[:, source_location[1]:source_location[1] + self.patch_size,
                 source_location[2]:source_location[2] + self.patch_size]

        target = self.noises_list[source_location[0]]
        target = target[:, source_location[1]:source_location[1] + self.patch_size,
                 source_location[2]:source_location[2] + self.patch_size]


        return source, target

    def __len__(self):
        return len(self.patches_list)

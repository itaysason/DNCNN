import random
import time
import scipy.misc
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch import optim
from torch.nn import MSELoss
from net import DNCNN
from test_utils import eval_func, show_plot, test_net_by_hand, move_data_to_variables_cuda, show_original_vs_denoised
from torch.utils.data import DataLoader
from dataset import DenoisedDataset
from general_utils import save_checkpoint, load_checkpoint

from PIL import Image, ImageOps
import os

random.seed(3000)

# settings
cuda = 1
transform = transforms.Compose([transforms.ToTensor()])
learning_rate = 0.0005
batch_size = 64
num_epochs = 50
num_images_to_plot = 1
train_net = True
load_net = False
net_path = 'denoiser_checkpoint.pth.tar'
best_model_path = 'best_score_model.pth.tar'
save_net = False
best_loss_score = float('inf')


# datasets definitions
train_noise_image_folder = dset.ImageFolder(root='data/train_data/noise')
train_images_image_folder = dset.ImageFolder(root='data/train_data/images')
train_dataset = DenoisedDataset(train_images_image_folder, train_noise_image_folder,
                                patch_size=60, stride=25, transform=transform, should_invert=False)
test_noise_image_folder = dset.ImageFolder(root='data/test_data/noise')
test_images_image_folder = dset.ImageFolder(root='data/test_data/images')
test_dataset = DenoisedDataset(test_images_image_folder, test_noise_image_folder,
                                patch_size=60, stride=25, transform=transform, should_invert=False)


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# defining net, loss and optimizer
l2_loss = MSELoss()
net = DNCNN(num_channels=64)

# check if saved file of model exists, if not continue without loading
if os.path.isfile(net_path) and load_net:
    best_loss_score = load_checkpoint(net, net_path)

# setting to cuda
if cuda >= 0:
    net.cuda(cuda)
    l2_loss.cuda(cuda)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

if train_net:

    # variables to keep the progress
    loss_history = []
    counter = []

    # training
    net.train()
    l2_loss.train()

    for epoch in range(num_epochs):
        tic = time.clock()
        # train routine - maybe i want to define a function for it
        for data in train_dataloader:

            source, target = move_data_to_variables_cuda(data, cuda)
            noise = net(source)
            optimizer.zero_grad()
            loss = l2_loss(noise, target)
            loss.backward()
            optimizer.step()

        print('epoch {} finished in {} seconds'.format(epoch, time.clock() - tic))
        if epoch % 10 == 0:
            test_net_by_hand(net, test_dataloader, l2_loss, num_images_to_plot, cuda)

        # adding current loss on validation data
        current_loss = eval_func(net, test_dataloader, l2_loss, cuda)
        loss_history.append(current_loss)
        counter.append(epoch)
        print('epoch {} loss is {}\n'.format(epoch, current_loss))

        # save progress
        if save_net:
            best_loss_score = save_checkpoint(net, best_loss_score, current_loss, net_path, best_model_path, cuda)


    # plot loss
    show_plot(counter, loss_history)
    test_net_by_hand(net, test_dataloader, l2_loss, num_images_to_plot, cuda)

    # in case we want test on whole particle
    particles_folder = dset.ImageFolder(root='data/full_particle_test')
    num_images = len(particles_folder.imgs)
    should_invert = False
    for i in range(num_images):
        img_tuple = particles_folder.imgs[i]
        particle_img = Image.open(img_tuple[0])
        particle_img = particle_img.convert("L")

        if should_invert:
            particle_img = ImageOps.invert(particle_img)

        if transform is not None:
            particle_img = transform(particle_img)

        particle_img = particle_img.resize_((1, 1, 127, 127))
        particle_img = move_data_to_variables_cuda([particle_img], cuda)[0]
        image_noise = net(particle_img)
        denoised_image = particle_img - image_noise
        show_original_vs_denoised(particle_img.cpu().data, denoised_image.cpu().data)
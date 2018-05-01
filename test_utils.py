import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable


def show_image_text(img1, img2):
    np_img1 = img1.numpy()[0, 0]
    np_img2 = img2.numpy()[0, 0]

    plt.subplot(1, 3, 1)
    plt.imshow(np.real(np_img1), cmap='gray')
    plt.title('Denoised')

    plt.subplot(1, 3, 2)
    plt.imshow(np.real(np_img2), cmap='gray')
    plt.title('Net Denoised')

    plt.subplot(1, 3, 3)
    plt.imshow(np.real(np_img1 - np_img2), cmap='gray')
    plt.title('difference')
    plt.show()


def show_original_vs_denoised(img1, img2):
    np_img1 = img1.numpy()[0, 0]
    np_img2 = img2.numpy()[0, 0]

    plt.subplot(1, 2, 1)
    plt.imshow(np.real(np_img1), cmap='gray')
    plt.title('Original')

    plt.subplot(1, 2, 2)
    plt.imshow(np.real(np_img2), cmap='gray')
    plt.title('Denoised')

    plt.show()


def show_plot(x_axis, y_axis):
    plt.plot(x_axis, y_axis)
    plt.show()


def eval_func(net, dataloader, loss_function, cuda):

    # check net and loss training status to return it to the way it was
    net_training_status = False
    if net.training:
        net_training_status = True
        net.eval()

    loss_training_status = False
    if loss_function.training:
        loss_training_status = True
        loss_function.eval()

    total_loss = 0
    for data in dataloader:

        image, noise = move_data_to_variables_cuda(data, cuda)
        output_noise = net(image)
        total_loss += loss_function(output_noise, noise).data[0] * image.shape[0]

    # return to the original status
    if net_training_status:
        net.train()
    if loss_training_status:
        loss_function.train()

    return total_loss


def test_net_by_hand(net, dataloader, loss_function, num_images, cuda):

    # check net and loss training status to return it to the way it was
    net_training_status = False
    if net.training:
        net_training_status = True
        net.eval()

    loss_training_status = False
    if loss_function.training:
        loss_training_status = True
        loss_function.eval()

    for i, data in enumerate(dataloader):
        if i >= num_images:
            break

        image, noise = move_data_to_variables_cuda(data, cuda)

        output_noise = net(image)

        # constructing original denoised image and the denoising by the net
        net_denoised_image = (image - output_noise)
        denoised_image = (image - noise)

        show_image_text(denoised_image.cpu().data, net_denoised_image.cpu().data)
        show_original_vs_denoised(image.cpu().data, net_denoised_image.cpu().data)

    # return to the original status
    if net_training_status:
        net.train()
    if loss_training_status:
        loss_function.train()


def move_data_to_variables_cuda(data, cuda):
    if cuda < 0:
        output = [Variable(var) for var in data]
    else:
        output = [Variable(var).cuda(cuda) for var in data]
    return tuple(output)

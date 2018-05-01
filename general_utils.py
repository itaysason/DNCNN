import torch
import shutil
import os


# Returns best loss score of the model till now. The model with this score saved in 'best_score_model.pth.tar'
def load_checkpoint(net, net_path='denoiser_checkpoint.pth.tar'):

    checkpoint = torch.load(net_path)
    net.load_state_dict(checkpoint['state_dict'])
    # In case we use only cuda (without specification) we can also save/load the optimizer values
    # In our case we can't do that because of the misfit between loading values and defined values
    # optimizer.load_state_dict(checkpoint['optimizer'])
    best_loss_score = checkpoint['best_loss_score']
    print("Checkpoint loaded successfully\n")
    return best_loss_score


def save_checkpoint(net, best_loss_score, current_loss_score, filename='denoiser_checkpoint.pth.tar',
                    best_model_path='best_score_model.pth.tar', cuda=1):

    best_score_changed = False

    if cuda >= 0:
        net.cpu()
    # We check if current loss is better than the best_loss_score
    if current_loss_score < best_loss_score:
        best_loss_score = current_loss_score
        best_score_changed = True

    # As mentioned before we can also save the optimizer values
    torch.save({'state_dict': net.state_dict(),
                # 'optimizer': optimizer,
                'best_loss_score': best_loss_score}, filename)

    print("Checkpoint saved successfully")
    # We will save to best model in case of best score or in case that there is no such a file
    if not(os.path.isfile(best_model_path)) or best_score_changed:
        shutil.copyfile(filename, best_model_path)
        print("Best model was updated to file path: {}\n".format(best_model_path))
    if cuda >= 0:
        net.cuda(cuda)

    return best_loss_score



from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt


def tensor_to_img(tensor):
    return ToPILImage()(tensor)


# Plots image tensor using matplotlib
def display(tensors, figsize=(6, 16)):
    plt.figure(figsize=figsize)
    for n, tensor in enumerate(tensors):
        img = tensor_to_img(tensor)
        sp = plt.subplot(1, len(tensors), n+1)
        sp.imshow(img)
        sp.axis('off')
    plt.show()

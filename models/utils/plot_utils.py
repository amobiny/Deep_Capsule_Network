import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import cv2
from config import args
import os


def visualize(images, conv_outputs, conv_grads, gb_vizs, probs, labels, img_size=28, fig_name='Hola'):

    num_imgs = images.shape[0]
    fig, axes = plt.subplots(nrows=4, ncols=num_imgs)
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.9, wspace=0.1, hspace=None)
    for ii, image, conv_output, conv_grad, gb_viz in zip(range(num_imgs), images, conv_outputs, conv_grads, gb_vizs):

        img_size = 51
        image = resize(image, (img_size, img_size, 1))
        gb_viz = resize(gb_viz, (img_size, img_size, 1))


        output = conv_output  # [7,7,512]
        grads_val = conv_grad  # [7,7,512]

        weights = np.mean(grads_val, axis=(0, 1))  # alpha_k, [512]
        cam = np.zeros(output.shape[0: 2], dtype=np.float32)  # [7,7]

        # Taking a weighted average
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]

        # Passing through ReLU
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)  # scale 0 to 1.0
        cam = resize(cam, (img_size, img_size), preserve_range=True)

        img = np.squeeze(image.astype(float))
        img -= np.min(img)
        img /= img.max()

        cam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam_heatmap = cv2.cvtColor(cam_heatmap, cv2.COLOR_BGR2RGB)

        ax = axes[0, ii]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        ax.set_title('{}, {}'.format(np.argmax(labels[ii]), np.argmax(probs[ii])))

        ax = axes[1, ii]
        ax.imshow(cam_heatmap)
        ax.axis('off')
        if ii == 0:
            ax.set_ylabel('Grad-CAM')

        ax = axes[2, ii]
        ax.imshow(img, cmap='gray')
        ax.imshow(cam_heatmap, alpha=0.4)
        ax.axis('off')
        if ii == 0:
            ax.set_ylabel('Grad-CAM overlay')

        gb_viz -= np.min(gb_viz)
        gb_viz /= gb_viz.max()

        ax = axes[3, ii]
        ax.imshow(img, cmap='gray')
        ax.imshow(np.squeeze(gb_viz) * cam, cmap='coolwarm', alpha=0.5)
        ax.axis('off')
        if ii == 0:
            ax.set_ylabel('guided Grad-CAM')

    FULL_NAME = os.path.join(args.imgdir+args.run_name, fig_name)
    width = 2 * num_imgs
    height = 8
    fig.set_size_inches(width, height)
    fig.savefig(FULL_NAME + '.png')
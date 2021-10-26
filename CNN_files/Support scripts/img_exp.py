import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import make_axes_locatable
import numpy as np
import imageio
import cv2 as cv
import os
from tqdm import tqdm
#model_names = next(os.walk("D:/Datasets/CNNs/old"))[1]
model_names = ["UNetResNet5lvl_resc_wrpd_10-14-2021_20-33-16_SSIMloss"]
img_nums = [3,32,56,120,178]

for model_name in tqdm(model_names):
    os.makedirs(rf"C:\Users\Michał\Desktop\DO RAPORTU\21.10\{model_name}",exist_ok=True)
    for img_num in img_nums:
        try:
            img2 = imageio.imread(f"D:/Datasets/CNNs/preds/{model_name}/pred_{img_num}.tiff").astype("float32")
            img = imageio.imread(f"D:/Datasets/dataset_8_Combined_to_pred_256x256_08-27-2021_12-17-33/resc_wrpd/{img_num}_resc_wrpd.tiff").astype("float32")
            img1 = imageio.imread(f"D:/Datasets/dataset_8_Combined_to_pred_256x256_08-27-2021_12-17-33/resc/{img_num}_resc.tiff").astype("float32")
            img = cv.resize(img, (512,512))
            img1 = cv.resize(img1, (512,512))
            img2 = cv.resize(img2, (512,512))

            def genlabels(n):
                label = rf"${int(n/np.pi)} \pi $"
                return label

            f, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True, figsize = (8,4))

            f.suptitle(f"Model: {model_name}.\nImage number: {img_num}.", y = 0.9)

            x = [0,256,512]
            y = [0,256,512]
            x_ticks = [0,128,256]
            y_ticks = [256,128,0]

            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            im1 = ax1.imshow(img, cmap='jet', aspect = 'equal')
            cbar_tick = [img.min(), img.max()]
            cbar_ticklabels = ['0', r'$2\pi$']
            cbar1 = f.colorbar(im1, ax=ax1, cax = cax1, ticks = cbar_tick)
            cbar1.ax.set_yticklabels(cbar_ticklabels)


            ax1.set_xticks(x)
            ax1.set_xticklabels(x_ticks)
            ax1.set_yticks(y)
            ax1.set_yticklabels(y_ticks)





            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            im2 = ax2.imshow(img1, cmap='jet', aspect = 'equal')
            cbar_tick = np.arange(img1.min(), img1.max(), 2*np.pi)
            cbar_ticklabels = list(map(genlabels,cbar_tick))
            cbar2 = f.colorbar(im2, ax=ax2, cax= cax2, ticks = cbar_tick)
            cbar2.ax.set_yticklabels(cbar_ticklabels)

            ax2.set_xticks(x)
            ax2.set_xticklabels(x_ticks)



            #cbar1.set_yticks(cbar_tick)



            divider3 = make_axes_locatable(ax3)
            cax3 = divider3.append_axes("right", size="5%", pad=0.05)
            im3 = ax3.imshow(img2, cmap='jet', aspect = 'equal')
            cbar_tick = np.arange(img2.min(), img2.max(), 2*np.pi)
            cbar_ticklabels = list(map(genlabels,cbar_tick))

            cbar3 = f.colorbar(im3, ax=ax3, cax=cax3, ticks = cbar_tick)
            cbar3.ax.set_yticklabels(cbar_ticklabels)

            ax3.set_xticks(x)
            ax3.set_xticklabels(x_ticks)



            ax1.title.set_text('Generated Wrapped')
            ax1.title.set_fontsize(10)
            ax2.title.set_text('Generated Unwrapped')
            ax2.title.set_fontsize(10)
            ax3.title.set_text('Prediction')
            ax3.title.set_fontsize(10)
            plt.tight_layout()
            plt.savefig(rf"C:\Users\Michał\Desktop\DO RAPORTU\21.10\{model_name}\{img_num}.png")
            #plt.show()
            plt.cla()
            plt.close(f)
        except FileNotFoundError: pass
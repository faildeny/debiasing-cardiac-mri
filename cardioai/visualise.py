import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

from cardioai.transforms import prepare_batch


def visualise(dataloader, config, directory="data_vis/"):
    TORCHIO_BACKEND = config["use_torchio"]

    if not os.path.exists(directory):
        os.makedirs(directory)

    for batch in dataloader:
        if TORCHIO_BACKEND:
            batch = prepare_batch(batch, config)
        for image, label, id, sample_weight, meta_label in zip(batch["image"], batch["label"], batch["id"], batch["sample_weight"], batch["meta_label"]):
            img = image
            # id = batch["id"][0]
            print(img.shape)
            # img = image.permute(2, 0, 1)

            if label[0] == 1:
                filename = f"Negative_{id}"
            else:
                filename = f"Positive_{id}"

            filename += f"_{meta_label}_weight_{sample_weight}"

            img = img.numpy()
            frames = []  # for storing the generated images
            # img = sample_gray.transpose(0, 3, 2, 1).astype(np.int16)
            fig = plt.figure()
            if len(img.shape) == 3 and img.shape[0] > 3:
                for i in range(img.shape[0]):
                    frames.append([plt.imshow(img[i], cmap=cm.Greys_r, animated=True)])

                ani = animation.ArtistAnimation(
                    fig, frames, interval=50, blit=True, repeat_delay=1000
                )
                ani.save(directory + filename + ".mp4")
            else:
                # plt.imshow(img, cmap=cm.Greys_r)
                img = img.transpose(1, 2, 0)
                plt.imshow(img)
                plt.savefig(directory + filename + ".png")

            # image = plt.imshow(image[0])
            # plt.savefig('image.png')

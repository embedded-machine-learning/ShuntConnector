import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curves(folder, fields_to_plot):

    fig, axs = plt.subplots(nrows=len(fields_to_plot), ncols=1, sharex=True)

    if type(axs) is not np.ndarray:
        axs = [axs]

    for i, field in enumerate(fields_to_plot):

        files = Path(folder).glob('*.npy')

        axs[i].grid()
        axs[i].set_ylabel(field)

        for file in files:
            if field == str(file.stem):
                axs[i].plot(np.load(file))
                break

        field = 'val_' + field
        for file in files:
            if field == str(file.stem):
                axs[i].plot(np.load(file))
                break
        
        axs[i].legend(['train', 'val'])
        

    plt.xlabel('Epochs')
    plt.show()


if __name__ == '__main__':

    if len(sys.argv) < 3:
        raise ValueError('Wrong number of arguments!')

    current_folder = sys.argv[1]
    fields_to_plot = sys.argv[2:]

    plot_learning_curves(current_folder, fields_to_plot)
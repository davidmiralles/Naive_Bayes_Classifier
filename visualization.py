
import matplotlib.pyplot as plt
import scipy.signal as sg
import seaborn as sn
from matplotlib.colors import Normalize
import numpy as np
class MidpointNormalize(Normalize):
    """
    Auxiliar function that helps on modifying the scale of colors of a matrix to allow a better visualization
    @Normalize: Normalize object
    """
    
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def _filter(values):
        #Filter parameters
        #order
        N = 6
        #sampling frequency
        fs = 40
        #cutoff frequency
        fc = 4
        #Wn = fc/(fs/2)
        Wn = fc/(fs/2)

        b, a = sg.butter(N, Wn, 'low')
        return sg.filtfilt(b, a, values)

def plot_accuracy_over_time(classifications, id_to_label, x_ticks, filtered=False):
    for y in classifications:
        target = id_to_label[int(y)]
        plt.figure(figsize=(10,7))
        plt.rcParams.update({'font.size': 32})
        plt.title(f'Accuracy {target}')
        
        for pred in range(len(classifications[y])):
            label = id_to_label[pred]
            if filtered:
                plt.plot(x_ticks, _filter(classifications[y][pred]), linewidth=5, label=label)
            else:
                plt.plot(x_ticks, classifications[y][pred], linewidth=5, label=label)

        plt.ylabel('Accuracy (%)')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.savefig(f'out/figures/{target}_filtered.png' if filtered else f'out/figures/{target}.png', bbox_inches='tight', dpi=600)
        plt.savefig(f'out/figures/{target}_filtered.svg' if filtered else f'out/figures/{target}.svg', bbox_inches='tight', dpi=600)
        plt.show()

def plot_confusion_matrix(confusion_matrix, id_to_label, n_samples=1000, n_tests=100):
    mean = np.trace(confusion_matrix)/confusion_matrix.shape[0]
    labels = [id_to_label[y] for y in range(confusion_matrix.shape[0])]

    plt.rcParams.update({'font.size': 10})
    plt.figure(figsize=(10,7))
    plt.suptitle('Confusion Matrix (%.2f%% accuracy, %d samples, %d tests)' % (mean, n_samples, n_tests), y=0.95, x=0.45, size=16)
    sn.set(font_scale=1.4)
    sn.heatmap(confusion_matrix, cmap='Blues', annot_kws={"size":12}, annot=True , xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted object")
    plt.ylabel("Actual object")
    plt.savefig(f'out/figures/confusion_matrix_{n_samples}samples_{n_tests}tests.png', bbox_inches='tight')
    plt.show()

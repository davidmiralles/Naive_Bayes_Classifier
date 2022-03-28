
import matplotlib.pyplot as plt
import scipy.signal as sg

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

def plot_accuracy_over_time(classifications, label_to_id, x_ticks, filtered=False):
    for y in classifications:
        target = list(label_to_id.keys())[list(label_to_id.values()).index(y)]
        plt.figure(figsize=(10,7))
        plt.rcParams.update({'font.size': 32})
        plt.title(f'Accuracy {target}')
        
        for pred in range(len(classifications[y])):
            label = list(label_to_id.keys())[list(label_to_id.values()).index(pred)]
            if filtered:
                plt.plot(x_ticks, _filter(classifications[y][pred]), linewidth=5, label=label)
            else:
                plt.plot(x_ticks, classifications[y][pred], linewidth=5, label=label)

        plt.ylabel('Accuracy (%)')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.savefig(f'out/img/{target}_filtered.png' if filtered else f'out/img/{target}.png', bbox_inches='tight', dpi=600)
        plt.savefig(f'out/img/{target}_filtered.svg' if filtered else f'out/img/{target}.svg', bbox_inches='tight', dpi=600)
        plt.show()
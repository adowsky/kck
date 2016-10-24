import matplotlib.pyplot as plotter
from matplotlib.markers import MarkerStyle
import matplotlib.lines as mlines

__author__ = 'Adrian Kosiński'


def avg(arr):
    return sum([float(x) for x in arr]) / float(len(arr))


def make_plot(lines, algorithm):
    x = []
    y = []
    for line in lines[1:]:
        splitted = line.split(',')
        x.append(float(splitted[1]) / 1000)
        y.append(avg(splitted[2:]) * 100)
    plotter.plot(x, y, algorithm['color'])


def plot_settings(algorithms):
    legend_entries = []
    for algorithm in algorithms:
        legend_entries.append(mlines.Line2D([], [], color=algorithm['color'], marker=algorithm['marker'],
                                            label=algorithm['name']))

    plotter.legend(handles=legend_entries, loc=4)

    plotter.xlabel('Rozegranych gier(x1000)')
    plotter.ylabel('Odsetek wygranych gier[%]')
    plotter.xlim([0, 500])
    plotter.ylim([60, 100])

    plotter.grid(True)


def make_scatter(lines, algorithm):
    x = []
    y = []
    for line in lines[1::37]:
        splitted = line.split(',')
        x.append(float(splitted[0]))
        y.append(avg(splitted[2:]) * 100)
    plotter.scatter(x, y, marker=MarkerStyle(algorithm['marker']), c=algorithm['color'], s=50)


def setting_scatter():
    plotter.xlabel('Populacja')
    plotter.xlim([0, 200])
    plotter.ylim([60, 100])


def make_boxplot_data(lines, algorithm):
    y = lines[-1].split(',')[2:]
    y = [float(x) * 100 for x in y]
    x = float(lines[-1].split(',')[1]) / 1000
    return {
        'x': x,
        'y': y,
        'label': algorithm['name']
    }


def make_boxplot(data):
    x = [element['x'] for element in data]
    y = [element['y'] for element in data]
    labels = [element['label'] for element in data]
    meanprops = dict(marker='o', markerfacecolor='blue')
    plotter.boxplot(y, x, showmeans=True, meanprops=meanprops)
    ticklabels = plotter.setp(plotter.gca(), xticklabels=labels)
    plotter.setp(ticklabels, rotation=45, fontsize=8)
    plotter.grid(True)
    plotter.gca().yaxis.tick_right()


def process_files(algorithms, processing_function, postprocessing=None):
    processing_results = []
    for algorithm in algorithms:
        with open(algorithm['file']) as data:
            lines = data.readlines()
            processing_results.append(processing_function(lines, algorithm))
    if postprocessing is not None:
        postprocessing(processing_results)


def main():
    plotter.subplot(1, 2, 1)

    algorithms = [
        dict(file='2cel.csv', marker='d', color='magenta', name='2-Coev'),
        dict(file='cel.csv', marker='s', color='black', name='1-Coev'),
        dict(file='2cel-rs.csv', marker='D', color='red', name='2-Coev-RS'),
        dict(file='cel-rs.csv', marker='v', color='green', name='1-Coev-RS'),
        dict(file='rsel.csv', marker='o', color='blue', name='1-Evol-RS')
    ]

    process_files(algorithms, make_plot)
    plot_settings(algorithms)

    # upper scale and points
    plotter.twiny()
    process_files(algorithms, make_scatter)
    setting_scatter()

    # second graph and boxplot
    plotter.subplot(1, 2, 2)
    process_files(algorithms, make_boxplot_data, make_boxplot)

    plotter.show()
    plotter.close()


if __name__ == '__main__':
    main()

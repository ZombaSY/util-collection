import pygal


class Subject:
    def __init__(self, disc_mean, color_mean, disc_dev, color_dev):
        self.disc_mean = disc_mean
        self.color_mean = color_mean
        self.disc_dev = disc_dev
        self.color_dev = color_dev


def insert_box_chart(title, box_chart, mean, dev):
    box_chart.add(title, ([mean, mean + dev, mean - dev]))


sub4 = Subject(
    [2.738095238, 2.19047619, 3.095238095, 1.976190476],
    [2.321428571, 2.321428571, 2.682926829, 2.678571429],
    [0.788595008, 1.029056323, 1.064794275, 1.14409655],
    [0.829668814, 1.138197345, 1.063565664, 0.924671297]
)

sub6 = Subject(
    [2.904761905, 2.511904762, 2.761904762, 1.821428571],
    [2.869047619, 2.30952381, 2.916666667, 1.904761905],
    [0.803549382, 1.037901358, 0.977351226, 1.045520083],
    [0.852583775, 1.023255664, 0.925284169, 1.047889575]
)

sub7 = Subject(
    [2.833333333, 2.476190476, 2.535714286, 2.154761905],
    [2.785714286, 2.642857143, 2.5, 2.071428571],
    [0.814062905, 1.128380534, 0.990387473, 1.115305293],
    [0.795394909, 1.108869621, 1.017700489, 1.09962888]
)

sub8 = Subject(
    [2.892857143, 2.464285714, 2.821428571, 1.821428571],
    [2.821428571, 2.571428571, 2.785714286, 1.821428571],
    [0.659506618, 1.06005872, 0.836812465, 1.233565431],
    [0.615488855, 1.083267921, 0.880630572, 1.233565431]
)

sub9 = Subject(
    [2.916666667, 2.19047619, 2.630952381, 2.261904762],
    [2.892857143, 2.214285714, 2.535714286, 2.357142857],
    [0.981475242, 0.96332992, 0.69057881, 1.086924972],
    [0.985222445, 0.939496174, 0.766818234, 1.21638474]
)

sub10 = Subject(
    [2.75, 2.571428571, 2.678571429, 2],
    [3.035714286, 2.428571429, 2.285714286, 2.25],
    [0.920985497, 0.979379229, 0.937457482, 1.210076739],
    [0.7897623, 1.032630878, 0.958314847, 1.221094357]
)


def render_box(subject, main_title):
    box_chart = pygal.Box()

    for i in range(4):
        if i == 0:
            title_mean = 'Input Disc'
            title_dev = 'Input Color'
        elif i == 1:
            title_mean = 'E-GAN Disc'
            title_dev = 'E-GAN Color'
        elif i == 2:
            title_mean = 'DeepLPF Disc'
            title_dev = 'DeepLPF Color'
        else:    # 3
            title_mean = 'CUD-NET Disc'
            title_dev = 'CUD-NET Color'

        insert_box_chart(title_mean, box_chart, subject.disc_mean[i], subject.disc_dev[i])
        insert_box_chart(title_dev, box_chart, subject.color_mean[i], subject.color_dev[i])

    box_chart.add('trash', [0.5, 4.3])  # set the threshold of output image.

    box_chart.render_to_file(main_title + '.svg')


render_box(sub4, 'sub4')
render_box(sub6, 'sub6')
render_box(sub7, 'sub7')
render_box(sub8, 'sub8')
render_box(sub9, 'sub9')
render_box(sub10, 'sub10')

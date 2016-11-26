import skimage
from skimage import data, filters, feature
from matplotlib import pylab as plt
from skimage.morphology import square


file_prefix = "samolot"
file_postfix = ".jpg"

view = []
fig = plt.figure()
for x in range(7, 13):
    file_no = "0" + str(x) if x < 10 else str(x)
    a = fig.add_subplot(2, 3, x-6)
    a.axes.get_xaxis().set_visible(False)
    a.axes.get_yaxis().set_visible(False)
    image = data.imread(file_prefix + file_no + file_postfix, as_grey=True)
    image = skimage.filters.rank.median(image, square(3))
    image =skimage.feature.canny(image, sigma=3, low_threshold=0.45)
    image = skimage.morphology.dilation(image, square(2))
    plt.imshow(image, cmap=plt.cm.gray)

fig.subplots_adjust(wspace=0, hspace=0)
plt.show()

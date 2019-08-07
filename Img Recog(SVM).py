import warnings

warnings.filterwarnings(action="ignore")
import matplotlib.pyplot as plt
from sklearn import datasets, svm       # datasets downloads the data of 2000 images

digits = datasets.load_digits()         # Loads the dict of 2000 images, first array is of 2000 imgs. second key
                                        # value holds the value of the imgs
print("\ndigits.keys() = ", digits.keys())

print("\n digits.target = ", digits.target)     # Prints the numbers

images_and_labels = list(zip(digits.images, digits.target))

print("\nlen(images_and_labels) = ", len(images_and_labels))

for index, [image, label] in enumerate(images_and_labels[:5]):
    print("\nindex : ", index, "image :\n ", image, "label : ", label)
    plt.subplot(2, 5, index + 1)
    plt.axis('on')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')  # It converts matrix into image,
    # cmap is the colour of the map Gray scaled , interpolation = 'nearest' to differentiate b/w 2 neighbouring pixels.

    plt.title("Training %i" % label)

# plt.show()

# To apply a classifier on this data, we need to flatten the image, to turn tha data into a (samples,features) matrix
n_samples = len(digits.images)
print("n_samples: ", n_samples)

imageData = digits.images.reshape((n_samples, -1))  # Reduce the dimension by 1 (8x8) converts to (1x64) for each image

print("After reshaped : len(imageData[0]) : ", len(imageData[0]))

# Create a classifier a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of digits
classifier.fit(imageData[: n_samples // 2], digits.target[: n_samples // 2])
# the shape of x becomes 1d digits.target is already 1d

expecty = digits.target[n_samples // 2:]
predy = classifier.predict(imageData[n_samples // 2:])

images_and_pred = list(zip(digits.images[n_samples // 2:], predy))

for index, [image, pred] in enumerate(images_and_pred[:5]):
    plt.subplot(2, 5, index + 6)
    plt.axis('on')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Predictions %i" % pred)

print("\nOriginal values : ", digits.target[n_samples // 2: (n_samples // 2) + 5])
plt.show()

from scipy.misc import imread, imresize, bytescale

img = imread("Three.jpeg")
img = imresize(img, (8, 8))         # Almost 1/10th reduction of image of the original image (80x80)
classfier = svm.SVC(gamma=0.001)
classifier.fit(imageData[:], digits.target[:])

print(digits.images.dtype)

img = img.astype(digits.images.dtype)   # Format of the two images must be same

img = bytescale(img, high=16.0, low=0)  # Scaling the testing image

print("\nimg.shape : ", img.shape)
print("\n", img)

xtestData = []

# the image ight be colured (r,g,b) so we merge the 3 image frames , that is every pixel is made of 3 colours
for c in img:
    for r in c:
        xtestData.append(sum(r) / 3.0)

print("\n xtestData : ", xtestData)
print("\nlen(xtestData) :", len(xtestData))

xtestData = [xtestData]     # 1x64
print("\nlen(xtestData) : ", len(xtestData))    # 1
print("\nMachine Output : ", classifier.predict(xtestData))
plt.show()

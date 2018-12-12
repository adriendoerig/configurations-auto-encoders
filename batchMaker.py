
# Class to make a batch

import numpy as np, random, matplotlib.pyplot as plt
from skimage import draw
from scipy.ndimage import zoom
from datetime import datetime
from parameters import *


def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top + h, trim_left:trim_left + w]

        # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def computeMeanAndStd(self, shapeTypes, batchSize=100, n_shapes=1, noiseLevel=0.0, max_rows=1, max_cols=3):
    # compute mean and std over a large batch. May be used to apply the same normalization to all images (cf. make_tf_dataset.py)

    batchImages = np.zeros(shape=(batchSize, self.imSize[0], self.imSize[1]), dtype=np.float32)
    batchSingleShapeImages = np.zeros(shape=(batchSize, self.imSize[0], self.imSize[1], n_shapes),
                                         dtype=np.float32)

    for n in range(batchSize):
        shapes = np.random.permutation(len(shapeTypes))[:n_shapes]

        for shape in range(n_shapes):
            if shapes[shape] == 0:  # 1/len(shapeTypes):
                thisOffset = random.randint(0, 1)
                batchSingleShapeImages[n, :, :, shape] = self.drawStim(False, shapeMatrix=[0], offset=thisOffset, offset_size=random.randint(1, int(self.barHeight / 2.0))) + np.random.normal(0, noiseLevel, size=self.imSize)
                batchImages[n, :, :] += batchSingleShapeImages[n, :, :, shape]

            else:
                thisType = shapes[shape]
                shapeType = shapeTypes[thisType]
                nRows = random.randint(1, max_rows)
                nCols = random.randint(1, max_cols)
                shapeConfig = shapeType * np.ones((nRows, nCols))
                batchSingleShapeImages[n, :, :, shape] = self.drawStim(0, shapeConfig) + np.random.normal(0, noiseLevel, size=self.imSize)
                batchImages[n, :, :] += batchSingleShapeImages[n, :, :, shape]

    batchMean = np.mean(batchImages)
    batchStd = np.std(batchImages)

    return batchMean, batchStd


class StimMaker:

    def __init__(self, imSize, shapeSize, barWidth):

        self.imSize    = imSize
        self.shapeSize = shapeSize
        self.barWidth  = barWidth
        self.barHeight = int(shapeSize/4-barWidth/4)
        self.offsetHeight = 1
        self.mean, self.std = computeMeanAndStd(self, shape_types, batchSize=100, n_shapes=simultaneous_shapes, noiseLevel=noise_level, max_rows=max_rows, max_cols=max_cols)


    def setShapeSize(self, shapeSize):

        self.shapeSize = shapeSize


    def drawSquare(self):

        resizeFactor = 1.2
        patch = np.zeros((self.shapeSize, self.shapeSize))

        firstRow = int((self.shapeSize - self.shapeSize/resizeFactor)/2)
        firstCol = firstRow
        sideSize = int(self.shapeSize/resizeFactor)

        patch[firstRow         :firstRow+self.barWidth,          firstCol:firstCol+sideSize+self.barWidth] = random.uniform(1-random_pixels, 1+random_pixels)
        patch[firstRow+sideSize:firstRow+self.barWidth+sideSize, firstCol:firstCol+sideSize+self.barWidth] = random.uniform(1-random_pixels, 1+random_pixels)
        patch[firstRow:firstRow+sideSize+self.barWidth, firstCol         :firstCol+self.barWidth         ] = random.uniform(1-random_pixels, 1+random_pixels)
        patch[firstRow:firstRow+sideSize+self.barWidth, firstRow+sideSize:firstRow+self.barWidth+sideSize] = random.uniform(1-random_pixels, 1+random_pixels)

        return patch


    def drawCircle(self):

        resizeFactor = 1.01
        radius = self.shapeSize/(2*resizeFactor)
        patch  = np.zeros((self.shapeSize, self.shapeSize))
        center = (int(self.shapeSize/2)-1, int(self.shapeSize/2)-1) # due to discretization, you maybe need add or remove 1 to center coordinates to make it look nice

        for row in range(self.shapeSize):
            for col in range(self.shapeSize):

                distance = np.sqrt((row-center[0])**2 + (col-center[1])**2)
                if radius-self.barWidth < distance < radius:
                    patch[row, col] = random.uniform(1-random_pixels, 1+random_pixels)

        return patch


    def drawPolygon(self, nSides, phi):

        resizeFactor = 1.0
        patch  = np.zeros((self.shapeSize, self.shapeSize))
        center = (int(self.shapeSize/2), int(self.shapeSize/2))
        radius = self.shapeSize/(2*resizeFactor)

        rowExtVertices = []
        colExtVertices = []
        rowIntVertices = []
        colIntVertices = []
        for n in range(nSides):
            rowExtVertices.append( radius               *np.sin(2*np.pi*n/nSides + phi) + center[0])
            colExtVertices.append( radius               *np.cos(2*np.pi*n/nSides + phi) + center[1])
            rowIntVertices.append((radius-self.barWidth)*np.sin(2*np.pi*n/nSides + phi) + center[0])
            colIntVertices.append((radius-self.barWidth)*np.cos(2*np.pi*n/nSides + phi) + center[1])

        RR, CC = draw.polygon(rowExtVertices, colExtVertices)
        rr, cc = draw.polygon(rowIntVertices, colIntVertices)
        patch[RR, CC] = random.uniform(1-random_pixels, 1+random_pixels)
        patch[rr, cc] = 0.0

        return patch


    def drawStar(self, nTips, ratio, phi):

        resizeFactor = 0.8
        patch  = np.zeros((self.shapeSize, self.shapeSize))
        center = (int(self.shapeSize/2), int(self.shapeSize/2))
        radius = self.shapeSize/(2*resizeFactor)

        rowExtVertices = []
        colExtVertices = []
        rowIntVertices = []
        colIntVertices = []
        for n in range(2*nTips):

            thisRadius = radius
            if not n%2:
                thisRadius = radius/ratio

            rowExtVertices.append(max(min( thisRadius               *np.sin(2*np.pi*n/(2*nTips) + phi) + center[0], self.shapeSize), 0.0))
            colExtVertices.append(max(min( thisRadius               *np.cos(2*np.pi*n/(2*nTips) + phi) + center[1], self.shapeSize), 0.0))
            rowIntVertices.append(max(min((thisRadius-self.barWidth)*np.sin(2*np.pi*n/(2*nTips) + phi) + center[0], self.shapeSize), 0.0))
            colIntVertices.append(max(min((thisRadius-self.barWidth)*np.cos(2*np.pi*n/(2*nTips) + phi) + center[1], self.shapeSize), 0.0))

        RR, CC = draw.polygon(rowExtVertices, colExtVertices)
        rr, cc = draw.polygon(rowIntVertices, colIntVertices)
        patch[RR, CC] = random.uniform(1-random_pixels, 1+random_pixels)
        patch[rr, cc] = 0.0

        return patch


    def drawIrreg(self, nSidesRough, repeatShape):

        if repeatShape:
            random.seed(1)

        patch  = np.zeros((self.shapeSize, self.shapeSize))
        center = (int(self.shapeSize/2-1), int(self.shapeSize/2-1))
        angle  = 0  # first vertex is at angle 0

        rowExtVertices = []
        colExtVertices = []
        rowIntVertices = []
        colIntVertices = []
        while angle < 2*np.pi:

            if np.pi/4 < angle < 3*np.pi/4 or 5*np.pi/4 < angle < 7*np.pi/4:
                radius = (random.random()+2.0)/3.0*self.shapeSize/2
            else:
                radius = (random.random()+1.0)/2.0*self.shapeSize/2

            rowExtVertices.append( radius               *np.sin(angle) + center[0])
            colExtVertices.append( radius               *np.cos(angle) + center[1])
            rowIntVertices.append((radius-self.barWidth)*np.sin(angle) + center[0])
            colIntVertices.append((radius-self.barWidth)*np.cos(angle) + center[1])

            angle += (random.random()+0.5)*(2*np.pi/nSidesRough)

        RR, CC = draw.polygon(rowExtVertices, colExtVertices)
        rr, cc = draw.polygon(rowIntVertices, colIntVertices)
        patch[RR, CC] = random.uniform(1-random_pixels, 1+random_pixels)
        patch[rr, cc] = 0.0

        if repeatShape:
            random.seed(datetime.now())

        return patch


    def drawStuff(self, nLines):

        patch  = np.zeros((self.shapeSize, self.shapeSize))

        for n in range(nLines):

            (r1, c1, r2, c2) = np.random.randint(self.shapeSize, size=4)
            rr, cc = draw.line(r1, c1, r2, c2)
            patch[rr, cc] = random.uniform(1-random_pixels, 1+random_pixels)

        return patch


    def drawVernier(self, offset=None, offset_size=None):

        if offset_size is None:
            offset_size = random.randint(1, int(self.barHeight/2.0))
        patch = np.zeros((2*self.barHeight+self.offsetHeight, 2*self.barWidth+offset_size))
        patch[0:self.barHeight, 0:self.barWidth] = 1.0
        patch[self.barHeight+self.offsetHeight:, self.barWidth+offset_size:] = random.uniform(1-random_pixels, 1+random_pixels)

        if offset is None:
            if random.randint(0, 1):
                patch = np.fliplr(patch)
        elif offset == 1:
            patch = np.fliplr(patch)

        fullPatch = np.zeros((self.shapeSize, self.shapeSize))
        firstRow  = int((self.shapeSize-patch.shape[0])/2)
        firstCol  = int((self.shapeSize-patch.shape[1])/2)
        fullPatch[firstRow:firstRow+patch.shape[0], firstCol:firstCol+patch.shape[1]] = patch

        return fullPatch


    def drawShape(self, shapeID, offset=None, offset_size=None):

        if shapeID == 0:
            patch = self.drawVernier(offset, offset_size)
        if shapeID == 1:
            patch = self.drawSquare()
        if shapeID == 2:
            patch = self.drawCircle()
        if shapeID == 3:
            patch = self.drawPolygon(6, 0)
        if shapeID == 4:
            patch = self.drawPolygon(8, np.pi/8)
        if shapeID == 5:
            patch = self.drawStar(4, 1.8, 0)
        if shapeID == 6:
            patch = self.drawStar(7, 1.7, -np.pi/14)
        if shapeID == 7:
            patch = self.drawIrreg(15, False)
        if shapeID == 8:
            patch = self.drawIrreg(15, True)
        if shapeID == 9:
            patch = self.drawStuff(5)
        if shapeID == 10:
            patch = self.drawPolygon(6, 0)

        return patch


    def drawStim(self, vernier, shapeMatrix, offset=None, offset_size=None, fixed_position=None):

        image        = np.zeros(self.imSize)
        critDist     = 0 # int(self.shapeSize/6)
        padDist      = 0 #int(self.shapeSize/6)
        shapeMatrix  = np.array(shapeMatrix)

        if len(shapeMatrix.shape) < 2:
            shapeMatrix = np.expand_dims(shapeMatrix, axis=0)

        if shapeMatrix.all() == None:  # this means we want only a vernier
            patch = np.zeros((self.shapeSize+5, self.shapeSize+5))
        else:
            patch = np.zeros((shapeMatrix.shape[0]*self.shapeSize + (shapeMatrix.shape[0]-1)*critDist + 1,
                                 shapeMatrix.shape[1]*self.shapeSize + (shapeMatrix.shape[1]-1)*critDist + 1))

            for row in range(shapeMatrix.shape[0]):
                for col in range(shapeMatrix.shape[1]):

                    firstRow = row*(self.shapeSize + critDist)
                    firstCol = col*(self.shapeSize + critDist)
                    patch[firstRow:firstRow+self.shapeSize, firstCol:firstCol+self.shapeSize] = self.drawShape(shapeMatrix[row,col], offset, offset_size)

        if vernier:

            firstRow = int((patch.shape[0]-self.shapeSize)/2) # + 1  # small adjustments may be needed depending on precise image size
            firstCol = int((patch.shape[1]-self.shapeSize)/2) # + 1
            patch[firstRow:firstRow+self.shapeSize, firstCol:firstCol+self.shapeSize] += self.drawVernier(offset, offset_size)
            patch[patch > 1.0] = 1.0

        if fixed_position is None:
            firstRow = random.randint(padDist, self.imSize[0] - (patch.shape[0]+padDist))  # int((self.imSize[0]-patch.shape[0])/2)
            firstCol = random.randint(padDist, self.imSize[1] - (patch.shape[1]+padDist))  # int((self.imSize[1]-patch.shape[1])/2)
        else:
            firstRow = fixed_position[0]
            firstCol = fixed_position[1]

        image[firstRow:firstRow+patch.shape[0], firstCol:firstCol+patch.shape[1]] = patch

        # make images with only -1 and 1
        image[image==0] = -0.
        image[image>0] = 1.

        return image


    def plotStim(self, vernier, shapeMatrix):

        plt.figure()
        plt.imshow(self.drawStim(vernier, shapeMatrix))
        plt.show()


    def makeConfigBatch(self, batchSize, configMatrix, doVernier = False, noiseLevel=0.0, normalize=False, normalize_sets=False, fixed_position=fixed_stim_position, vernierLabelEncoding='nothinglr_012', random_size=False):

        batchImages   = np.ndarray(shape=(batchSize, self.imSize[0], self.imSize[1]), dtype=np.float32)
        vernierLabels = np.zeros(batchSize, dtype=np.float32)

        for n in range(batchSize):

            offset = random.randint(0, 1)
            batchImages[n, :, :] = self.drawStim(doVernier, shapeMatrix=configMatrix, fixed_position=fixed_position, offset=offset) + np.random.normal(0, noiseLevel, size=self.imSize)
            if normalize:
                # batchImages[batchImages < 0] = 0
                # batchImages[batchImages > 0.2] = 1
                batchImages[n, :, :] = (batchImages[n, :, :] - np.mean(batchImages[n, :, :])) / np.std(batchImages[n, :, :])**vernier_normalization_exp
            if vernierLabelEncoding is 'nothinglr_012':
                vernierLabels[n] = -offset + 2
            elif vernierLabelEncoding is 'lr_01':
                vernierLabels[n] = -offset + 1

            if random_size:
                zoom_factor = random.uniform(0.8, 1.2)
                tempImage = clipped_zoom(batchImages[n, :, :], zoom_factor)
                tempImage[tempImage == 0] = -np.mean(tempImage)  # because when using random_sizes, small images get padded with 0 but the background may be <= because of normalization
                if tempImage.shape == batchImages[n, :, :].shape:
                    batchImages[n, :, :] = tempImage

        if normalize_sets:
            batchImages = (batchImages - self.mean) / self.std

        batchImages = np.expand_dims(batchImages, -1)  # need to a a fourth dimension for tensorflow

        return batchImages, vernierLabels


    def showBatch(self, batchSize, config, noiseLevel=0.0, normalize=False, fixed_position=fixed_stim_position, vernierLabelEncoding='lr_01'):

        # input a configuration to display
        batchImages, batchLabels = self.makeConfigBatch(batchSize, config, doVernier=False, noiseLevel=noiseLevel, normalize=normalize, fixed_position=fixed_position, vernierLabelEncoding=vernierLabelEncoding)

        for n in range(batchSize):
            plt.figure()
            plt.imshow(batchImages[n, :, :, 0])
            plt.title('Label, mean, stdev = ' + str(batchLabels[n]) + ', ' + str(
                np.mean(batchImages[n, :, :, 0])) + ', ' + str(np.std(batchImages[n, :, :, 0])))
            plt.show()



if __name__ == "__main__":

    other_shape_ID = 7  # there will be squares and this shape in the array
    rufus = StimMaker((32, 52), 10, 1)
    stim_matrix = np.random.randint(2, size=(3, 5))*(other_shape_ID-1) + 1
    stim_matrix[1, 2] = 1

    # rufus.plotStim(1, [[1, 2, 3], [4, 5, 6], [6, 7, 0]])
    rufus.showBatch(1, stim_matrix, noise_level)

import rasterfairy

#xy should be a numpy array with a shape (number of points,2)
grid_xy = rasterfairy.transformPointCloud2D(xy)
#grid_xy will contain the points in the same order but aligned to a grid

from rioxarray import open_rasterio

all_data = open_rasterio('data/tiff/ArDS.tif')
data = all_data[-48:]
data.rio.to_raster('data/validate/reference.tif')
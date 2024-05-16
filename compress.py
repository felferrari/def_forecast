from pathlib import Path
from osgeo_utils import gdal_merge


path_files = Path(r'D:\Ferrari\Projects\deforestation-forescast\exp-1km\resunet_output\true')

out_file = path_files / 'reference-resunet.tif'
out_file.unlink(missing_ok=True)
list_files = sorted(list(path_files.glob('*.tif')))
list_files = [str(file) for file in list_files]

parameters = ['', '-separate', '-o', str(out_file)] + list_files
gdal_merge.main(parameters)
print(list_files)
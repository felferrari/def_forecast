import argparse
from pathlib import Path
from rioxarray import open_rasterio
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shutil import rmtree
from rasterio.enums import Resampling
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--reference', type = Path)
parser.add_argument('--predictions', type = Path)
parser.add_argument('--mask', type = Path)
#parser.add_argument('--output', type = Path)
parser.add_argument('--output_figures', type = Path)
#parser.add_argument('--max_cells', type = int)
#parser.add_argument('--threshold', type = float)
parser.add_argument('--cell_size', type = float)

args = parser.parse_args()

def main():
    #open geotiffs
    reference = open_rasterio(args.reference)
    predictions = open_rasterio(args.predictions)
    mask = open_rasterio(args.mask)
    
    assert reference.shape == predictions.shape, 'Reference and Predictions must have the same dimensions'
    
    # threshold = args.threshold
    cell_size = args.cell_size
    max_cells = 10 #args.max_cells
    # output_file = args.output
    # output_file.unlink(missing_ok=True)
    output_figures = args.output_figures
    if output_figures.exists():
        rmtree(output_figures)
    output_figures.mkdir()
    
    
    #Construct the base geopandas    
    # x, y = reference.x, reference.y
    # x, y = np.meshgrid(x, y)
    # x, y = x.flatten(), y.flatten()
    # geo = gpd.GeoSeries.from_xy(x = x, y=y)
    # geo = geo.buffer(cell_size/2.0, cap_style= 3)
    # base_gpd = gpd.GeoDataFrame(geometry=geo, crs = reference.rio.crs)
    # base_gpd['mask'] = mask.values.flatten()
    
    # #Create the perdictions shapefile    
    # pred_gpd = base_gpd.copy()
    # ref_gpd = base_gpd.copy()
    # y0, m0, d0 = 22, 1, 1
    # biweeks = []
    # dates = []
    # for i in range(48):
    #     y = y0 + i // 24
    #     m = m0 + ((i // 2) % 12)
    #     d = d0 + 15 * (i % 2)
    #     biweeks.append(f'PREV{d:02d}{m:02d}{y:02d}')
    #     dates.append(f'{d:02d}-{m:02d}-{y:02d}')
    #     pred_gpd[f'PREV{d:02d}{m:02d}{y:02d}'] = predictions.values[i].flatten()
    #     ref_gpd[f'REF{d:02d}{m:02d}{y:02d}'] = reference.values[i].flatten()
    
    # pred_gpd = pred_gpd.drop(pred_gpd[pred_gpd['mask'] == 0].index)
    # ref_gpd = ref_gpd.drop(ref_gpd[ref_gpd['mask'] == 0].index)
    # pred_gpd.to_file(output_file, layer = 'predictions', driver="GPKG")
    # ref_gpd.to_file(output_file, layer = 'reference', driver="GPKG")
    
    mask_48 = reference.copy()
    mask_48.values = np.repeat(mask, 48, axis=0)
    
    non_zeros = predictions.copy()
    non_zeros.values = np.logical_or(
        predictions.values > 0,
        reference.values > 0,
    ).astype(np.float32)
    
    norm_predictions = predictions.copy()
    norm_predictions.values[mask_48==0] = np.nan
    pred_mins, pred_maxs = np.nanmin(norm_predictions.values, axis=(1,2), keepdims= True), np.nanmax(norm_predictions.values, axis=(1,2), keepdims= True)
    norm_predictions.values = (norm_predictions.values - pred_mins) / (pred_maxs - pred_mins)

    norm_reference = reference.copy()
    norm_reference.values[mask_48==0] = np.nan
    ref_mins, ref_maxs = np.nanmin(norm_reference.values, axis=(1,2), keepdims= True), np.nanmax(norm_reference.values, axis=(1,2), keepdims= True)
    norm_reference.values = (norm_reference.values - ref_mins) / (ref_maxs - ref_mins)
    
    absolute_error = norm_predictions.copy()
    absolute_error.values = norm_predictions.values - norm_reference.values
    absolute_error.rio.write_nodata(np.nan, inplace = True)
    
    absolute_error.rio.to_raster(output_figures / f'absolute_error.tif')
    norm_predictions.rio.to_raster(output_figures / f'predictions.tif')
    norm_reference.rio.to_raster(output_figures / f'reference.tif')
    
    means_zeros_in = []
    means_non_zeros = []
    
    for downscale_factor in range(1, max_cells+ 1):
        
        absolute_error_sampled = absolute_error.rio.reproject(
            absolute_error.rio.crs,
            resolution=(downscale_factor*cell_size, downscale_factor*cell_size),
            resampling=Resampling.average,
        )
        
        means_zeros_in.append(np.nanmean(absolute_error_sampled.values, axis=(1,2)))
        absolute_error_sampled.rio.to_raster(output_figures / f'absolute_error_zerosin_r{downscale_factor}.tif')
        
        non_zeros_sampled = non_zeros.rio.reproject(
            non_zeros.rio.crs,
            resolution=(downscale_factor*cell_size, downscale_factor*cell_size),
            resampling=Resampling.average,
        )
        
        absolute_error_sampled.values[non_zeros_sampled.values == 0] = np.nan
        means_non_zeros.append(np.nanmean(absolute_error_sampled.values, axis=(1,2)))
        
        
        
    means_zeros_in = np.stack(means_zeros_in)
    df = pd.DataFrame(
        data = means_zeros_in
    )
    df.to_excel(output_figures / 'zeros_in_results.xlsx')
    
    means_non_zeros = np.stack(means_non_zeros)
    df = pd.DataFrame(
        data = means_non_zeros
    )
    df.to_excel(output_figures / 'non_zeros_results.xlsx')
    
if __name__ == '__main__':
    
    
    main()
    
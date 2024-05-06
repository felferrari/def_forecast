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
parser.add_argument('--output', type = Path)
parser.add_argument('--output_figures', type = Path)
parser.add_argument('--max_cells', type = int)
parser.add_argument('--threshold', type = float)
parser.add_argument('--cell_size', type = float)

args = parser.parse_args()

def main():
    #open geotiffs
    reference = open_rasterio(args.reference)
    predictions = open_rasterio(args.predictions)
    mask = open_rasterio(args.mask)
    
    assert reference.shape == predictions.shape, 'Reference and Predictions must have the same dimensions'
    
    threshold = args.threshold
    cell_size = args.cell_size
    max_cells = args.max_cells
    output_file = args.output
    output_file.unlink(missing_ok=True)
    output_figures = args.output_figures
    if output_figures.exists():
        rmtree(output_figures)
    output_figures.mkdir()
    
    
    #Construct the base geopandas    
    x, y = reference.x, reference.y
    x, y = np.meshgrid(x, y)
    x, y = x.flatten(), y.flatten()
    geo = gpd.GeoSeries.from_xy(x = x, y=y)
    geo = geo.buffer(cell_size/2.0, cap_style= 3)
    base_gpd = gpd.GeoDataFrame(geometry=geo, crs = reference.rio.crs)
    base_gpd['mask'] = mask.values.flatten()
    
    #Create the perdictions shapefile    
    pred_gpd = base_gpd.copy()
    ref_gpd = base_gpd.copy()
    y0, m0, d0 = 22, 1, 1
    biweeks = []
    dates = []
    for i in range(48):
        y = y0 + i // 24
        m = m0 + ((i // 2) % 12)
        d = d0 + 15 * (i % 2)
        biweeks.append(f'PREV{d:02d}{m:02d}{y:02d}')
        dates.append(f'{d:02d}-{m:02d}-{y:02d}')
        pred_gpd[f'PREV{d:02d}{m:02d}{y:02d}'] = predictions.values[i].flatten()
        ref_gpd[f'REF{d:02d}{m:02d}{y:02d}'] = reference.values[i].flatten()
    
    pred_gpd = pred_gpd.drop(pred_gpd[pred_gpd['mask'] == 0].index)
    ref_gpd = ref_gpd.drop(ref_gpd[ref_gpd['mask'] == 0].index)
    pred_gpd.to_file(output_file, layer = 'predictions', driver="GPKG")
    ref_gpd.to_file(output_file, layer = 'reference', driver="GPKG")
    
    #predictions.values[predictions.values < 0.01] = 0
    
    absolute_error = predictions - reference
    relative_error = np.abs(absolute_error) / reference
    
    mask_ = np.repeat(mask, 48, axis=0)
    
    absolute_error.values[mask_.values == 0] = 0
    relative_error.values[mask_.values == 0] = 0
    
    absolute_error.rio.to_raster(output_figures / 'absolute_error.tif')
    relative_error.rio.to_raster(output_figures / 'relative_error.tif')
    
    similarity = (relative_error * 100) < threshold
    similarity.values = similarity.values.astype(np.float32)
    similarity.values[mask_.values == 0] = np.nan
    similarity.rio.write_nodata(np.nan, inplace = True)
    similarity.rio.to_raster(output_figures / f'similarity_th{threshold}.tif')
    
    #Consider similarity = 0 when predictions ==0 and reference == 0
    similarity_zeroin = similarity.copy()
    similarity_zeroin.values[np.logical_and(predictions.values ==0, reference.values == 0)] = 1
    similarity_zeroin.values[mask_.values == 0] = np.nan
    similarity_zeroin.rio.write_nodata(np.nan, inplace = True)
    similarity_zeroin.rio.to_raster(output_figures / f'similarity_zeroin_th{threshold}.tif')
    
    
    corrected_relative_error = relative_error.copy().values
    corrected_relative_error[np.logical_and(predictions.values ==0, reference.values == 0)] = 0
    corrected_relative_error = corrected_relative_error.flatten()
    corrected_relative_error = corrected_relative_error[mask_.values.flatten() == 1]
    corrected_relative_error = corrected_relative_error[corrected_relative_error != np.inf]
    corrected_relative_error = 100*corrected_relative_error
    corrected_relative_error = np.clip(corrected_relative_error, 0, 1000)
    
    fig, ax = plt.subplots()
    plt.hist(corrected_relative_error, bins = 50, log=True)
    plt.xlabel('%')
    plt.ylabel('Count')
    fig.savefig(output_figures / f'relative_errors_hist_0.png')
    plt.close(fig)
    
    corrected_relative_error = np.clip(corrected_relative_error, 0, 100)
    fig, ax = plt.subplots()
    plt.hist(corrected_relative_error, bins = 50, log=True)
    plt.xlabel('%')
    plt.ylabel('Count')
    fig.savefig(output_figures / f'relative_errors_hist_1.png')
    plt.close(fig)
    
    corrected_relative_error = np.clip(corrected_relative_error, 0, 10)
    fig, ax = plt.subplots()
    plt.hist(corrected_relative_error, bins = 50, log=True)
    plt.xlabel('%')
    plt.ylabel('Count')
    fig.savefig(output_figures / f'relative_errors_hist_2.png')
    plt.close(fig)
    
    corrected_relative_error = np.clip(corrected_relative_error, 0, 1)
    fig, ax = plt.subplots()
    plt.hist(corrected_relative_error, bins = 50, log=True)
    plt.xlabel('%')
    plt.ylabel('Count')
    fig.savefig(output_figures / f'relative_errors_hist_3.png')
    plt.close(fig)
    
    # priority cells
    shape = reference.shape[1:]
    similarity_priority = predictions.copy()
    similarity_priority.values = np.zeros_like(similarity_priority)
    pred_order = predictions.copy()
    pred_order.values = np.zeros_like(pred_order)
    ref_order = predictions.copy()
    ref_order.values = np.zeros_like(ref_order)
    diff_order = predictions.copy()
    diff_order.values = np.zeros_like(diff_order)
    for biweek in range(predictions.shape[0]):
        pred_bw = predictions.values[biweek].flatten()
        ref_bw = reference.values[biweek].flatten()
        
        pred_order_bw = np.zeros_like(pred_bw)
        ref_order_bw = np.zeros_like(ref_bw)
        
        pred_order_bw[np.argsort(pred_bw)[-100:]] = (np.arange(100, 0, -1) - 1)
        ref_order_bw[np.argsort(ref_bw)[-100:]] =  (np.arange(100, 0, -1) - 1)
        
        diff_bw = np.abs(pred_order_bw - ref_order_bw)
        
        diff_order.values[biweek] = diff_bw.reshape(shape)
        ref_order.values[biweek] = ref_order_bw.reshape(shape)
        pred_order.values[biweek] = pred_order_bw.reshape(shape)
        
        diff_bw = (diff_bw <= 10).astype(np.float32).reshape(shape)
        similarity_priority.values[biweek] = diff_bw
        
    similarity_priority.values[mask_.values == 0] = np.nan
    similarity_priority.rio.write_nodata(np.nan, inplace = True)
    similarity_priority.rio.to_raster(output_figures / f'similarity_priority.tif')
    
    diff_order.values[mask_.values == 0] = np.nan
    diff_order.rio.write_nodata(np.nan, inplace = True)
    diff_order.rio.to_raster(output_figures / f'diff_order.tif')
    
    ref_order.values[mask_.values == 0] = np.nan
    ref_order.rio.write_nodata(np.nan, inplace = True)
    ref_order.rio.to_raster(output_figures / f'ref_order.tif')
    
    pred_order.values[mask_.values == 0] = np.nan
    pred_order.rio.write_nodata(np.nan, inplace = True)
    pred_order.rio.to_raster(output_figures / f'pred_order.tif')
    
    
    means = []
    means_nonzeros = []
    means_priority = []
    
    for downscale_factor in range(1, max_cells+ 1):
        
        similarity_sampled = similarity_zeroin.rio.reproject(
            similarity_zeroin.rio.crs,
            resolution=(downscale_factor*cell_size, downscale_factor*cell_size),
            resampling=Resampling.average,
        )

        similarity_sampled.rio.to_raster(output_figures / f'similarity_r{downscale_factor}.tif')
        
        means.append(100*np.nanmean(similarity_sampled, axis=(1,2)))
        
        similarity_sampled = similarity.rio.reproject(
            similarity.rio.crs,
            resolution=(downscale_factor*cell_size, downscale_factor*cell_size),
            resampling=Resampling.average,
        )

        similarity_sampled.rio.to_raster(output_figures / f'similarity_non_zeros_r{downscale_factor}.tif')
        
        means_nonzeros.append(100*np.nanmean(similarity_sampled, axis=(1,2)))
        
        similarity_sampled = similarity_priority.rio.reproject(
            similarity_priority.rio.crs,
            resolution=(downscale_factor*cell_size, downscale_factor*cell_size),
            resampling=Resampling.average,
        )

        similarity_sampled.rio.to_raster(output_figures / f'similarity_priority_r{downscale_factor}.tif')
        
        means_priority.append(100*np.nanmean(similarity_sampled, axis=(1,2)))
        
    means = np.stack(means)
    df = pd.DataFrame(
        data = means
    )
    df.to_excel(output_figures / 'results.xlsx')
    
    means_nonzeros = np.stack(means_nonzeros)
    df = pd.DataFrame(
        data = means_nonzeros
    )
    df.to_excel(output_figures / 'results_nonzeros.xlsx')
    
    means_priority = np.stack(means_priority)
    df = pd.DataFrame(
        data = means_priority
    )
    df.to_excel(output_figures / 'results_priority.xlsx')
    
    
    
    
if __name__ == '__main__':
    
    
    main()
    

        # minx, miny, maxx, maxy = similarity_sampled.rio.bounds()
    
    # add_x = (downscale_factor - (similarity_zeroin.rio.width % downscale_factor)) % downscale_factor
    # add_y = (downscale_factor - (similarity_zeroin.rio.height % downscale_factor)) % downscale_factor
    
    # #add_x += downscale_factor
    # #add_y += downscale_factor
    
    # maxx += cell_size * add_x
    # maxy += cell_size * add_y
    
    # similarity_sampled = similarity_sampled.rio.pad_box(
    #     minx = minx,
    #     miny = miny,
    #     maxx= maxx,
    #     maxy= maxy,
    #     constant_values = np.nan
    # )
    
    # new_width = similarity_sampled.rio.width // downscale_factor
    # new_height = similarity_sampled.rio.height // downscale_factor
    
    # a = block_reduce(
    #     similarity_sampled.values, 
    #     block_size=(1, downscale_factor, downscale_factor)
    #     )

    # similarity_sampled = similarity_sampled.rio.reproject(
    #     similarity_sampled.rio.crs,
    #     shape=(new_height, new_width),
    #     resampling=Resampling.average,
    # )
    
    # similarity_sampled.values = block_reduce(
    #     similarity.values,
    #     (1, downscale_factor, downscale_factor),
    #     np.nanmean
    # )
    
        # mask_sampled =  similarity_zeroin.copy()
        # mask_sampled.values = mask_.values.astype(np.float32)
        # mask_sampled = mask_sampled.rio.reproject(
        #     similarity_zeroin.rio.crs,
        #     resolution=(downscale_factor*cell_size, downscale_factor*cell_size),
        #     resampling=Resampling.sum,
        # )
        # similarity_sampled.values = similarity_sampled.values / mask_sampled.values
        
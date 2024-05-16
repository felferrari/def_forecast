import argparse
from pathlib import Path
from rioxarray import open_rasterio
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shutil import rmtree
from rasterio.enums import Resampling
import pandas as pd
from einops import rearrange

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
    
    top_k = 100
    
    n_bands = reference.shape[0]
    
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
    means_priority_cells = []
    
    real_predictions = predictions.copy()
    real_predictions.values[mask_48==0] = np.nan
    real_predictions.rio.write_nodata(np.nan, inplace = True)
    
    real_reference = reference.copy()
    real_reference.values[mask_48==0] = np.nan
    real_reference.rio.write_nodata(np.nan, inplace = True)
    
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
        absolute_error_non_zeros_sampled = absolute_error_sampled.copy()
        absolute_error_non_zeros_sampled.values[non_zeros_sampled.values == 0] = np.nan
        means_non_zeros.append(np.nanmean(absolute_error_non_zeros_sampled.values, axis=(1,2)))
        
        
        real_predictions_sampled = real_predictions.rio.reproject(
            real_predictions.rio.crs,
            resolution=(downscale_factor*cell_size, downscale_factor*cell_size),
            resampling=Resampling.sum,
        )
        
        real_reference_sampled = real_reference.rio.reproject(
            real_reference.rio.crs,
            resolution=(downscale_factor*cell_size, downscale_factor*cell_size),
            resampling=Resampling.sum,
        )
        
        pred_mins, pred_maxs = np.nanmin(real_predictions_sampled.values, axis=(1,2), keepdims= True), np.nanmax(real_predictions_sampled.values, axis=(1,2), keepdims= True)
        ref_mins, ref_maxs = np.nanmin(real_reference_sampled.values, axis=(1,2), keepdims= True), np.nanmax(real_reference_sampled.values, axis=(1,2), keepdims= True)
        real_predictions_sampled.values = (real_predictions_sampled.values - pred_mins) / (pred_maxs - pred_mins)
        real_reference_sampled.values = (real_reference_sampled.values - ref_mins) / (ref_maxs - ref_mins)
        
        real_predictions_sampled.rio.to_raster(output_figures / f'norm_prediction_r{downscale_factor}.tif')
        real_reference_sampled.rio.to_raster(output_figures / f'norm_reference_r{downscale_factor}.tif')
        
        # pred_flat = rearrange(real_predictions_sampled.values, 'n h w -> n (h w)')
        # ref_flat = rearrange(real_reference_sampled.values, 'n h w -> n (h w)')
        
        
        # order_pred_flat = np.zeros_like(pred_flat)
        # order_pred_flat[np.argsort(pred_flat, axis=-1)[:top_k]] = np.arange(top_k, 0, -1)
        order_pred, order_ref = [], []
        
        shape = real_predictions_sampled.shape[1:]
        
        for band_i in range(n_bands):
            pred_flat_i = real_predictions_sampled.values[band_i].flatten()
            ref_flat_i = real_reference_sampled.values[band_i].flatten()
            
            pred_flat_i = np.nan_to_num(pred_flat_i, 0)
            ref_flat_i = np.nan_to_num(ref_flat_i, 0)
        
            order_pred_flat_i = np.zeros_like(pred_flat_i)
            order_ref_flat_i = np.zeros_like(ref_flat_i)
            
            # order_pred_flat_i[np.argsort(pred_flat_i)[-top_k:]] = np.arange(100, 0, -1)
            order_pred_flat_i[np.argsort(pred_flat_i)[-top_k:]] = np.arange(1, 101)
            # order_ref_flat_i[np.argsort(ref_flat_i)[-top_k:]] = np.arange(100, 0, -1)
            order_ref_flat_i[np.argsort(ref_flat_i)[-top_k:]] = np.arange(1, 101)
            
            order_pred.append(order_pred_flat_i.reshape(shape))
            order_ref.append(order_ref_flat_i.reshape(shape))
            
        order_predictions_sampled = real_predictions_sampled.copy()
        order_predictions_sampled.values = np.stack(order_pred, axis=0)
        
        order_reference_sampled = real_reference_sampled.copy()
        order_reference_sampled.values = np.stack(order_ref, axis=0)
        
        order_predictions_sampled.rio.to_raster(output_figures / f'order_prediction_r{downscale_factor}.tif')
        order_reference_sampled.rio.to_raster(output_figures / f'order_reference_r{downscale_factor}.tif')
        
        order_diff_sampled = order_predictions_sampled.copy()
        order_diff_sampled.values = order_predictions_sampled.values - order_reference_sampled.values
        order_diff_sampled.values[np.isnan(absolute_error_sampled.values)] = np.nan
        order_diff_sampled.rio.to_raster(output_figures / f'order_real_difference_r{downscale_factor}.tif')
        order_diff_sampled.values = np.abs(order_diff_sampled.values)
        order_diff_sampled.rio.to_raster(output_figures / f'order_abs_difference_r{downscale_factor}.tif')
        
        means_priority_cells.append(np.nanmean(order_diff_sampled.values, axis=(1,2)))

        
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
    
    means_priority_cells = np.stack(means_priority_cells)
    df = pd.DataFrame(
        data = means_priority_cells
    )
    df.to_excel(output_figures / 'prio_cells_results.xlsx')
    
if __name__ == '__main__':
    
    
    main()
    
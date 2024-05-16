import argparse
from pathlib import Path
from rioxarray import open_rasterio
import numpy as np
from shutil import rmtree
from rasterio.enums import Resampling
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--reference', type = Path)
parser.add_argument('--predictions', type = Path)
parser.add_argument('--mask', type = Path)
parser.add_argument('--output_figures', type = Path)
parser.add_argument('--cell_size', type = float)

args = parser.parse_args()

def normalize(data):
    """
    The `normalize` function takes a 3D array of data, calculates the minimum and maximum values along
    the specified axes, and normalizes the data accordingly.
    
    @param data It seems like you have not provided the actual data that you want to normalize. Please
    provide the data so that I can assist you in normalizing it using the given function.
    
    @return The function `normalize` takes a 3-dimensional array `data`, calculates the minimum and
    maximum values along the last two dimensions, and then normalizes the data using these values. The
    normalized data is stored in a new array `norm_data` which is a copy of the input data. Finally, the
    function returns the normalized data `norm_data`.
    """
    norm_data = data.copy()
    
    mins, maxs = np.nanmin(norm_data.values, axis=(1,2), keepdims= True), np.nanmax(norm_data.values, axis=(1,2), keepdims= True)
    norm_data.values = (norm_data.values - mins) / (maxs - mins)
    
    return norm_data
    
def sort_data(data, top_k):
    """
    The function `sort_data` takes a multi-dimensional array of data, sorts each band based on the top k
    values, and returns the sorted data.
    
    @param data It seems like you were about to provide the `data` parameter for the `sort_data`
    function, but it got cut off. Could you please provide the `data` parameter so that I can assist you
    further with sorting the data?
    @param top_k The `top_k` parameter in the `sort_data` function represents the number of top elements
    to select and rank within each band of the input data. This parameter determines how many elements
    will be considered in the sorting process for each band.
    
    @return The function `sort_data` returns a new data object where the values have been sorted based
    on the top_k values in each band.
    """
    bands = data.shape[0]
    shape = data.shape[1:]
    
    data_l = []    
    for band in range(bands):
        data_i_flat = data.values[band].flatten()
        data_i_flat = np.nan_to_num(data_i_flat, 0)
        #order_flat_i = np.zeros_like(data_i_flat)
        order_flat_i = (top_k+1)*np.ones_like(data_i_flat)
        order_flat_i[np.argsort(data_i_flat)[-top_k:]] = np.arange(top_k, 0, -1)
        data_l.append(order_flat_i.reshape(shape))
        
    order_data = data.copy()
    order_data.values = np.stack(data_l, axis=0)
    
    return order_data

def top_k_classification(data, top_k):
    """
    The function `top_k_classification` takes a multi-band image data and returns a new data array with
    the top k values in each band marked as 1 and the rest as 0.
    
    @param data The `data` parameter in the `top_k_classification` function seems to be a
    multi-dimensional array or a DataFrame with shape (bands, *shape), where `bands` represents the
    number of bands and `shape` represents the shape of each band's data.
    @param top_k The `top_k` parameter in the `top_k_classification` function represents the number of
    top values to select from each band of the input data for classification. This parameter determines
    how many top values will be marked as 1 in the output data for each band.
    
    @return The function `top_k_classification` returns a modified version of the input data with the
    top k values in each band set to 1 and all other values set to 0.
    """
    bands = data.shape[0]
    shape = data.shape[1:]
    
    data_l = []    
    for band in range(bands):
        data_i_flat = data.values[band].flatten()
        data_i_flat = np.nan_to_num(data_i_flat, 0)
        order_flat_i = np.zeros_like(data_i_flat)
        order_flat_i[np.argsort(data_i_flat)[-top_k:]] = 1
        data_l.append(order_flat_i.reshape(shape))
        
    order_data = data.copy()
    order_data.values = np.stack(data_l, axis=0)
    
    return order_data

def save_means(means, fname):
    """
    The function `save_means` saves a list of means to an Excel file.
    
    @param means It looks like you were about to provide some information about the `means` parameter,
    but it seems to have been cut off. Could you please provide more details about the `means` parameter
    so that I can assist you further with the `save_means` function?
    @param fname The `fname` parameter in the `save_means` function is a string that represents the file
    name or path where the DataFrame containing the means will be saved as an Excel file.
    """
    means = np.stack(means)
    df = pd.DataFrame(
        data = means
    )
    df.to_excel(fname)

def main():
    #open geotiffs
    reference = open_rasterio(args.reference)
    predictions = open_rasterio(args.predictions)
    mask = open_rasterio(args.mask)
    
    assert reference.shape == predictions.shape, 'Reference and Predictions must have the same dimensions'
    
    cell_size = args.cell_size
    max_downscale = 10 #args.max_cells
    output_figures = args.output_figures
    if output_figures.exists():
        rmtree(output_figures)
    output_figures.mkdir()
    
    top_k = 100
    
    n_bands = reference.shape[0]
    
    mask_3d = reference.copy()
    mask_3d.values = np.repeat(mask, n_bands, axis=0)
    
    real_predictions = predictions.copy()
    real_predictions.values[mask_3d==0] = np.nan
    real_predictions.rio.write_nodata(np.nan, inplace = True)
    
    real_reference = reference.copy()
    real_reference.values[mask_3d==0] = np.nan
    real_reference.rio.write_nodata(np.nan, inplace = True)
    
    means_diff, means_abs_diff, means_nonzeros_diff, means_abs_nonzeros_diff, means_priority2, means_abs_priority = [], [], [], [], [], []
    
    for downscale_factor in range(1, max_downscale + 1):
        
        #Real Error
        predictions_sampled = real_predictions.rio.reproject(
            real_predictions.rio.crs,
            resolution=(downscale_factor*cell_size, downscale_factor*cell_size),
            resampling=Resampling.sum,
        )
        
        reference_sampled = real_reference.rio.reproject(
            real_reference.rio.crs,
            resolution=(downscale_factor*cell_size, downscale_factor*cell_size),
            resampling=Resampling.sum,
        )
        
        predictions_sampled = normalize(predictions_sampled)
        reference_sampled = normalize(reference_sampled)
        
        diff_sampled = predictions_sampled.copy()
        diff_sampled.values = predictions_sampled.values - reference_sampled.values
        diff_sampled.rio.to_raster(output_figures / f'error_d{downscale_factor}.tif')
        
        means_diff.append(np.nanmean(diff_sampled.values, axis=(1,2)))
        means_abs_diff.append(np.nanmean(np.abs(diff_sampled.values), axis=(1,2)))
        
        #Non Zeros Error
        non_zeros_predictions_sampled = real_predictions.rio.reproject(
            real_predictions.rio.crs,
            resolution=(downscale_factor*cell_size, downscale_factor*cell_size),
            resampling=Resampling.sum,
        )
        
        non_zeros_reference_sampled = real_reference.rio.reproject(
            real_reference.rio.crs,
            resolution=(downscale_factor*cell_size, downscale_factor*cell_size),
            resampling=Resampling.sum,
        )
        
        zeros_zeros = np.logical_and(
            non_zeros_predictions_sampled.values == 0, 
            non_zeros_reference_sampled.values == 0, 
        )
        
        non_zeros_predictions_sampled.values[zeros_zeros] = np.nan
        non_zeros_predictions_sampled.rio.write_nodata(np.nan, inplace = True)
        non_zeros_reference_sampled.values[zeros_zeros] = np.nan
        non_zeros_reference_sampled.rio.write_nodata(np.nan, inplace = True)
        
        non_zeros_predictions_sampled = normalize(non_zeros_predictions_sampled)
        non_zeros_reference_sampled = normalize(non_zeros_reference_sampled)
        
        non_zeros_diff_sampled = non_zeros_predictions_sampled.copy()
        non_zeros_diff_sampled.values = non_zeros_predictions_sampled.values - non_zeros_reference_sampled.values
        non_zeros_diff_sampled.rio.to_raster(output_figures / f'error_nonzeros_d{downscale_factor}.tif')
        
        means_nonzeros_diff.append(np.nanmean(non_zeros_diff_sampled.values, axis=(1,2)))
        means_abs_nonzeros_diff.append(np.nanmean(np.abs(non_zeros_diff_sampled.values), axis=(1,2)))
        
        # Priority Cells
        predictions_ordered = sort_data(predictions_sampled, top_k)
        reference_ordered = sort_data(reference_sampled, top_k)
        
        predictions_ordered.values[np.isnan(diff_sampled.values)] = np.nan
        predictions_ordered.rio.write_nodata(np.nan, inplace = True)
        
        reference_ordered.values[np.isnan(diff_sampled.values)] = np.nan
        reference_ordered.rio.write_nodata(np.nan, inplace = True)
        
        predictions_ordered.rio.to_raster(output_figures / f'predictions_order_d{downscale_factor}.tif')
        reference_ordered.rio.to_raster(output_figures / f'reference_order_d{downscale_factor}.tif')
        
        diff_order = predictions_ordered.copy()
        diff_order.values[np.isnan(diff_sampled.values)] = np.nan
        diff_order.rio.write_nodata(np.nan, inplace = True)
        diff_order.values = predictions_ordered.values - reference_ordered.values
        diff_order.rio.to_raster(output_figures / f'difference_order_d{downscale_factor}.tif')
        diff_order.values = np.abs(predictions_ordered.values - reference_ordered.values)
        diff_order.rio.to_raster(output_figures / f'difference_abs_order_d{downscale_factor}.tif')
        #means_priority.append(np.nanmean(diff_order.values, axis=(1,2)))
        means_abs_priority.append(np.nanmean(np.abs(diff_order.values), axis=(1,2)))
        
        top_ks = []
        top_k_values = []
        top_k_total = []
        max_k = 100
        for k in range(1, max_k+1):
            top_k_predictions = top_k_classification(predictions_sampled, k)
            top_k_reference = top_k_classification(reference_sampled, k)
            
            matches = np.logical_and(
                top_k_predictions.values == 1,
                top_k_reference.values == 1,
            )
            
            top_ks.append(k)
            top_k_values.append(matches.sum(axis=(1,2))/(k))
            top_k_total.append(matches.sum()/(k))
        
        results = pd.DataFrame(data={
            'k': top_ks,
            'values': top_k_total
        })
        
        results2 = pd.DataFrame(data=top_k_values)
        means_priority2.append(list(results2.sum()/max_k))
        
        
    save_means(means_diff, output_figures / 'diff_results.xlsx')
    save_means(means_abs_diff, output_figures / 'diff_abs_results.xlsx')
    save_means(means_nonzeros_diff, output_figures / 'diff_nonzeros_results.xlsx')
    save_means(means_abs_nonzeros_diff, output_figures / 'diff_abs_nonzeros_results.xlsx')
    save_means(means_priority2, output_figures / 'priority2_results.xlsx')
    save_means(means_abs_priority, output_figures / 'priority_abs_results.xlsx')
        
    
if __name__ == '__main__':
    
    
    main()
    
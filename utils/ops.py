import json
import numpy as np
from osgeo import gdal_array
from osgeo import gdal, gdalconst
from pathlib import Path
from typing import Union
import yaml 
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from einops import rearrange, repeat
import mlflow
import matplotlib
import matplotlib.animation as animation
import tempfile
import collections.abc
from collections.abc import MutableMapping
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from tqdm import tqdm
from matplotlib.transforms import Bbox
import pandas as pd

def flatten_dict(dictionary, parent_key='', separator='_'):
    """
    The `flatten_dict` function recursively flattens a nested dictionary into a single-level dictionary
    with keys concatenated using a separator.
    
    @param dictionary The `dictionary` parameter in the `flatten_dict` function is the input dictionary
    that you want to flatten. This dictionary can contain nested dictionaries as values. The function
    recursively flattens this dictionary into a single-level dictionary where the keys are concatenated
    with the parent keys using the specified separator.
    @param parent_key The `parent_key` parameter in the `flatten_dict` function is used to keep track of
    the current key hierarchy while recursively flattening a nested dictionary. It represents the key of
    the parent dictionary in the current recursive call.
    @param separator The `separator` parameter in the `flatten_dict` function is used to separate keys
    in the flattened dictionary. By default, the separator is set to `'_'`, but you can change it to any
    other character or string if you prefer a different separator.
    
    @return The function `flatten_dict` returns a flattened dictionary where nested keys are combined
    using the specified separator.
    """
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def deepupdate(d, u):
    """
    The `deepupdate` function recursively updates a dictionary with the values from another dictionary.
    
    @param d The `d` parameter in the `deepupdate` function is a dictionary that will be updated with
    the values from the `u` dictionary. The function recursively updates the values in `d` with the
    values from `u`, going deep into nested dictionaries if necessary.
    @param u The `u` parameter in the `deepupdate` function is a dictionary containing the updates that
    need to be applied to the original dictionary `d`. It iterates over the key-value pairs in `u` and
    updates the corresponding key in `d` with the new value. If the value is
    
    @return The `deepupdate` function is returning the dictionary `d` after performing deep update with
    the dictionary `u`. It recursively updates the values of `d` with the values from `u`, and returns
    the updated dictionary `d`.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deepupdate(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def remove_outliers(img, signficance = 0.01):
    """
    The function `remove_outliers` takes an image as input and clips the pixel values based on the
    specified significance level to remove outliers.
    
    @param img The `img` parameter in the `remove_outliers` function is expected to be a NumPy array
    representing an image. The function calculates the lower and upper quantiles for each color channel
    of the image and clips the pixel values to remove outliers based on the specified significance
    level.
    @param signficance The `significance` parameter in the `remove_outliers` function determines the
    proportion of outliers to be removed from the image. It is used to calculate the quantiles that
    define the range within which pixel values are considered non-outliers. The default value for
    `significance` is set to
    
    @return The function `remove_outliers` returns the input image `img` with outliers removed based on
    the specified significance level.
    """
    outliers = np.quantile(img, [signficance, 1-signficance], axis = (0,1))
    for channel in range(img.shape[-1]):
        img[:,:,channel] = np.clip(img[:,:,channel], outliers[0, channel],  outliers[1, channel])
    return img


def load_json(fp):
    """
    The function `load_json` reads and loads a JSON file from the given file path.
    
    @param fp The `fp` parameter in the `load_json` function stands for the file path to the JSON file
    that you want to load and read. It is used to specify the location of the JSON file on your system.
    
    @return The function `load_json(fp)` is returning the content of the JSON file located at the file
    path `fp`.
    """
    with open(fp) as f:
        return json.load(f)
    
def save_json(dict_:dict, file_path: Union[str, Path]) -> None:
    """
    The function `save_json` saves a dictionary as a JSON file with an optional indentation of 4 spaces.
    
    @param dict_ The `dict_` parameter is a dictionary that contains the data you want to save as a JSON
    file.
    @param file_path The `file_path` parameter in the `save_json` function is the path to the file where
    the JSON data will be saved. It can be either a string representing the file path or a `Path` object
    from the `pathlib` module. This parameter specifies the location where the JSON data
    """
    with open(file_path, 'w') as f:
        json.dump(dict_, f, indent=4)

def save_yaml(dict_:dict, file_path: Union[str, Path]) -> None:
    """
    The function `save_yaml` saves a dictionary to a YAML file at the specified file path.
    
    @param dict_ The `dict_` parameter is a dictionary that contains the data you want to save to a YAML
    file.
    @param file_path The `file_path` parameter is a string or a `Path` object that represents the path
    to the file where the YAML data will be saved. It specifies the location and name of the file where
    the YAML data will be written.
    """
    with open(file_path, 'w') as f:
        yaml.dump(dict_, f, default_flow_style=False)

def load_yaml(file_path: Union[str, Path]) -> None:
    """
    The function `load_yaml` reads and loads a YAML file from the specified file path.
    
    @param file_path The `file_path` parameter in the `load_yaml` function is a string or a `Path`
    object that represents the path to the YAML file that you want to load and parse.
    
    @return The function `load_yaml` is supposed to load and parse a YAML file, but it is currently
    returning the result of `yaml.safe_load(f)`. However, the function is annotated to return `None`. To
    fix this, you should update the return type hint of the function to match the actual return value,
    which is the parsed YAML data.
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
    
def load_opt_image(img_file):
    """
    The function `load_opt_image` loads an image file, converts it to a float16 array, handles NaN
    values, adjusts the shape if necessary, and then normalizes the values before returning the image.
    
    @param img_file The `img_file` parameter in the `load_opt_image` function is expected to be a file
    path to an image file that will be loaded and processed.
    
    @return The function `load_opt_image` returns a preprocessed image array. The image is loaded from
    the specified file using GDAL library, converted to float16 data type, and any NaN values are
    replaced with 0. If the image is 2-dimensional, it is expanded to have a third dimension. Finally,
    the image array is transposed to have the channel dimension as the last axis and normalized
    """
    img = gdal_array.LoadFile(str(img_file)).astype(np.float16)
    img[np.isnan(img)] = 0
    if len(img.shape) == 2 :
        img = np.expand_dims(img, axis=0)
    return np.moveaxis(img, 0, -1) / 10000

def load_sb_image(img_file):
    """
    The function `load_sb_image` loads an image file using GDAL and returns the image.
    
    @param img_file The `img_file` parameter in the `load_sb_image` function is a string that represents
    the file path of an image file that you want to load using GDAL (Geospatial Data Abstraction
    Library).
    
    @return The function `load_sb_image` is returning the image data loaded from the specified image
    file using GDAL library.
    """
    img = gdal_array.LoadFile(str(img_file))
    return img

def get_nodata(img_file):
    srs = gdal.Open(str(img_file))
    return srs.GetRasterBand(1).GetNoDataValue()

def load_ml_image(img_file):
    """
    This function loads a multi-band image file, handles NaN values, and adjusts the image dimensions if
    necessary before returning the image array.
    
    @param img_file The `img_file` parameter is a file path to an image file that will be loaded using
    GDAL (Geospatial Data Abstraction Library) and converted to a NumPy array of type float32. The
    function then handles any NaN (Not a Number) values in the image array by setting
    
    @return The function `load_ml_image` returns a NumPy array representing the image data with the
    following characteristics:
    - Any NaN values in the image data are replaced with 0.
    - If the image data is 2-dimensional, it is expanded to have a third dimension.
    - The dimensions of the array are rearranged using `np.moveaxis` so that the first dimension becomes
    the last dimension.
    """

    img = gdal_array.LoadFile(str(img_file)).astype(np.float32)
    img[np.isnan(img)] = 0
    if len(img.shape) == 2 :
        img = np.expand_dims(img, axis=0)
    return np.moveaxis(img, 0, -1)


def load_SAR_image(img_file):
    """
    The function `load_SAR_image` loads a Synthetic Aperture Radar (SAR) image file, processes it by
    setting NaN values to 0, and then reorders the axes of the image array.
    
    @param img_file The `img_file` parameter in the `load_SAR_image` function is a file path to the SAR
    (Synthetic Aperture Radar) image that you want to load and process. This function uses GDAL
    (Geospatial Data Abstraction Library) to load the image data from the specified
    
    @return The function `load_SAR_image` is returning the SAR image data after loading the image file,
    replacing any NaN values with 0, and then rearranging the axes of the image array. The final output
    is the SAR image data with the axes moved from the first position to the last position.
    """

    img = gdal_array.LoadFile(str(img_file))
    #img = 10**(img/10) 
    img[np.isnan(img)] = 0
    return np.moveaxis(img, 0, -1)

def save_geotiff(base_image_path, dest_path, data, dtype, nodata = None):
    """
    This function saves a NumPy array as a GeoTIFF file using GDAL library in Python.
    
    @param base_image_path The `base_image_path` parameter in the `save_geotiff` function is the file
    path to the base image that will be used as a reference for the geospatial information in the new
    GeoTIFF file.
    @param dest_path The `dest_path` parameter in the `save_geotiff` function is the destination path
    where the GeoTIFF file will be saved. It should be a string representing the full path including the
    filename and extension of the output GeoTIFF file.
    @param data It looks like the code snippet you provided is a function for saving a GeoTIFF file
    using GDAL library in Python. The function takes several parameters including `base_image_path`,
    `dest_path`, `data`, `dtype`, and `nodata`.
    @param dtype The `dtype` parameter in the `save_geotiff` function specifies the data type of the
    input data that you want to save as a GeoTIFF file. It can take on the following values:
    @param nodata The `nodata` parameter in the `save_geotiff` function is used to specify a pixel value
    that should be treated as NoData in the output GeoTIFF file. This means that any pixel in the input
    data that has a value equal to the specified `nodata` value will
    """
    base_image_path = str(base_image_path)
    base_data = gdal.Open(base_image_path, gdalconst.GA_ReadOnly)

    geo_transform = base_data.GetGeoTransform()
    x_res = base_data.RasterXSize
    y_res = base_data.RasterYSize
    crs = base_data.GetSpatialRef()
    proj = base_data.GetProjection()
    dest_path = str(dest_path)

    if len(data.shape) == 2:
        if dtype == 'byte':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, 1, gdal.GDT_Byte)
            data = data.astype(np.uint8)
        elif dtype == 'float':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, 1, gdal.GDT_Float32)
            data = data.astype(np.float32)
        elif dtype == 'uint16':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, 1, gdal.GDT_UInt16)
            data = data.astype(np.uint16)
    elif len(data.shape) == 3:
        if dtype == 'byte':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, data.shape[-1], gdal.GDT_Byte)
            data = data.astype(np.uint8)
        elif dtype == 'float':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, data.shape[-1], gdal.GDT_Float32)
            data = data.astype(np.float32)
        elif dtype == 'float':
            target_ds = gdal.GetDriverByName('GTiff').Create(dest_path, x_res, y_res, data.shape[-1], gdal.GDT_UInt16)
            data = data.astype(np.uint16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetSpatialRef(crs)
    target_ds.SetProjection(proj)

    if len(data.shape) == 2:
        target_ds.GetRasterBand(1).WriteArray(data)
        if nodata is not None:
            target_ds.GetRasterBand(1).SetNoDataValue(nodata)
    elif len(data.shape) == 3:
        for band_i in range(1, data.shape[-1]+1):
            target_ds.GetRasterBand(band_i).WriteArray(data[:,:,band_i-1])
            if nodata is not None:
                target_ds.GetRasterBand(1).SetNoDataValue(nodata)
    target_ds = None

def count_parameters_old(model):
    """Count the number of model parameters
    Args:
        model (Module): Model
    Returns:
        int: Number of Model's parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_parameters(model):
    total_params = 0
    text = ''
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params+=params
        text += f'{name}: {params:,}\n'
    text+=f'Total: {total_params:,}\n'
    return total_params

def evaluate_bins(ref, pred, bins, metric):
    """
    The function `evaluate_bins` calculates a specified metric for different bins of values in two
    arrays.
    
    @param ref The `ref` parameter in the `evaluate_bins` function is typically a NumPy array containing
    the reference values for evaluation. It is used to create bins based on the values in this array for
    further analysis and comparison with the predicted values.
    @param pred It seems like you were about to provide the `pred` parameter for the `evaluate_bins`
    function. Please go ahead and provide the values for the `pred` parameter so that I can assist you
    further.
    @param bins Bins is a list of values that define the boundaries for binning the data. The function
    `evaluate_bins` will evaluate the performance of the model predictions based on these bins.
    @param metric The `metric` parameter in the `evaluate_bins` function is a function that calculates a
    performance metric between the reference values (`ref_bin`) and predicted values (`pred_bin`) for
    each bin. This metric could be any evaluation measure such as mean squared error, accuracy,
    precision, recall, F1
    
    @return The function `evaluate_bins` returns a dictionary where the keys represent the bins and the
    values represent the result of applying the specified metric function to the corresponding reference
    and prediction values within each bin.
    """
    bin_value_0 = bins[0]
    ref_bin = ref[ref == bin_value_0]
    pred_bin = pred[ref == bin_value_0]
    res__dict = {
        str(bins[0]): metric(ref_bin, pred_bin)
    }
    for i, bin_value_end in enumerate(bins):
        if i == 0:
            continue
        ref_bin = ref[np.logical_and(ref > bin_value_0, ref <= bin_value_end)]
        pred_bin = pred[np.logical_and(ref > bin_value_0, ref <= bin_value_end)]
        res__dict [f'{bin_value_0}-{bin_value_end}'] = metric(ref_bin, pred_bin)
        bin_value_0 = bin_value_end
    return res__dict

def evaluate_results(reference, predictions, mask, bins = [0, 100], run_name = ''):
    """
    This Python function evaluates metrics, generates histograms, and visualizes results for reference
    and prediction data, including handling normalization and time series analysis.
    
    :param reference: The `reference` parameter in the `evaluate_results` function is typically a NumPy
    array representing the ground truth or actual values that you are comparing your predictions
    against. It is used to calculate metrics such as Mean Squared Error (MSE) and Mean Absolute Error
    (MAE) between the reference
    :param predictions: The `predictions` parameter in the `evaluate_results` function is typically a
    NumPy array containing the predicted values for a certain task or model. It is used to compare the
    predicted values with the reference values to evaluate the performance of the model. The function
    calculates various metrics such as Mean Squared Error
    :param mask: The `mask` parameter in the `evaluate_results` function is used to specify a mask that
    filters out certain values from the `reference` and `predictions` arrays during evaluation. The mask
    is applied to flatten the arrays and calculate metrics only on the values that are not filtered out
    by the mask
    :param bins: The `bins` parameter in the `evaluate_results` function is a list that specifies the
    bin edges for histogram binning. By default, it is set to `[0, 100]`, which means the histogram will
    have bins with edges at 0 and 100. You can customize this parameter
    :return: The function `evaluate_results` returns the following values:
    - `mse`: Mean Squared Error between the reference and prediction data
    - `mae`: Mean Absolute Error between the reference and prediction data
    - `norm_mse`: Normalized Mean Squared Error between the normalized reference and prediction data
    - `norm_mae`: Normalized Mean Absolute Error between the normalized reference and prediction data
    """
    #evaluate metrics
    plt.style.use('seaborn-v0_8-deep')
    original_shape = reference.shape[:2]
    ref_flatten_mask = rearrange(reference, 'h w c -> (h w) c')[mask.flatten() == 1]
    pred_flatten_mask = rearrange(predictions, 'h w c -> (h w) c')[mask.flatten() == 1]
    mse = mean_squared_error(ref_flatten_mask, pred_flatten_mask)
    mae = mean_absolute_error(ref_flatten_mask, pred_flatten_mask)
    
    matplotlib.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(14, 6))
    plt.hist([ref_flatten_mask.flatten(), pred_flatten_mask.flatten()], label = ['Reference', 'Prediction'], color = ['red', 'blue'],bins = 80, log=True, rwidth = 0.9, range = [0, 80])
    plt.title('Prediction Histogram')
    plt.legend(loc='upper right')
    mlflow.log_figure(fig, f'figures/hist_{run_name}.png')
    plt.close(fig)
    #metrics with bins
    
    mse__dict = evaluate_bins(ref_flatten_mask, pred_flatten_mask, bins, mean_squared_error)
    mae__dict = evaluate_bins(ref_flatten_mask, pred_flatten_mask, bins, mean_absolute_error)
    
    matplotlib.rcParams.update({'font.size': 12})
    fig, ax= plt.subplots(figsize = (10, 6))
    bar = plt.bar(range(len(mse__dict)), list(mse__dict.values()), align='center')
    plt.xticks(range(len(mse__dict)), list(mse__dict.keys()))
    plt.xlabel('Reference Values Bins (Km2)')
    ax.bar_label(bar, fmt='{:,.3f}')
    plt.ylabel('MSE')
    plt.ylim([0,600])
    mlflow.log_figure(fig, f'figures/bar_mse_{run_name}.png')
    plt.close(fig)
    
    fig, ax= plt.subplots(figsize = (10, 6))
    bar = plt.bar(range(len(mae__dict)), list(mae__dict.values()), align='center')
    plt.xticks(range(len(mae__dict)), list(mae__dict.keys()))
    plt.xlabel('Reference Values Bins (Km2)')
    ax.bar_label(bar, fmt='{:,.3f}')
    plt.ylabel('MAE')
    plt.ylim([0,20])
    mlflow.log_figure(fig, f'figures/bar_mae_{run_name}.png')
    plt.close(fig)
    
    
    fig, ax= plt.subplots(figsize = (10, 6))
    bar = plt.bar(range(len(mse__dict)), list(mse__dict.values()), align='center')
    plt.xticks(range(len(mse__dict)), list(mse__dict.keys()))
    plt.xlabel('Reference Values Bins (Km2)')
    plt.ylabel('MSE (Log)')
    ax.set_yscale('log')
    ax.bar_label(bar, fmt='{:,.3f}')
    plt.ylim([1e-4,5000])
    mlflow.log_figure(fig, f'figures/bar_log_mse_{run_name}.png')
    plt.close(fig)
    
    fig, ax= plt.subplots(figsize = (10, 6))
    bar = plt.bar(range(len(mae__dict)), list(mae__dict.values()), align='center')
    plt.xticks(range(len(mae__dict)), list(mae__dict.keys()))
    plt.xlabel('Reference Values Bins (Km2)')
    plt.ylabel('MAE (Log)')
    ax.set_yscale('log')
    ax.bar_label(bar, fmt='{:,.3f}')
    plt.ylim([1e-3,50])
    mlflow.log_figure(fig, f'figures/bar_log_mae_{run_name}.png')
    plt.close(fig)
    
    #metrics for each lag and normalization
    mse_list, mae_list = [],[]
    norm_mse_list, norm_mae_list = [],[]
    norm_ref, norm_preds = np.empty_like(reference), np.empty_like(predictions)
    for i in range(ref_flatten_mask.shape[-1]):
        ref_i = ref_flatten_mask[:,i]
        pred_i = pred_flatten_mask[:,i]
        
        mse_list.append(mean_squared_error(ref_i, pred_i))
        mae_list.append(mean_absolute_error(ref_i, pred_i))
        
        ref_i = reference[:,:,i]
        pred_i = predictions[:,:,i]
        
        ref_i_flatten_mask = ref_i.flatten()[mask.flatten() == 1]
        pred_i_flatten_mask = pred_i.flatten()[mask.flatten() == 1]
        
        if ref_i_flatten_mask.max() != ref_i_flatten_mask.min():
            ref_i_norm = (ref_i - ref_i_flatten_mask.min())/(ref_i_flatten_mask.max() - ref_i_flatten_mask.min())
        else:
            ref_i_norm = np.zeros_like(ref_i)
        if pred_i_flatten_mask.max() != pred_i_flatten_mask.min():
            pred_i_norm = (pred_i - pred_i_flatten_mask.min())/(pred_i_flatten_mask.max() - pred_i_flatten_mask.min())
        else:
            pred_i_norm = np.zeros_like(pred_i)
            
        ref_i_norm[mask == 0] = 0
        pred_i_norm[mask == 0] = 0
        
        ref_i[mask == 0] = 0
        pred_i[mask == 0] = 0
            
        norm_ref[:,:,i] = ref_i_norm
        norm_preds[:,:,i] = pred_i_norm
        
        fig, axarr = plt.subplots(1,2)
        im_0 = axarr[0].imshow(ref_i, cmap = 'gray', vmin = ref_i_flatten_mask.min(), vmax = ref_i_flatten_mask.max())
        axarr[0].axis("off")
        axarr[0].set_title("Reference")
        divider = make_axes_locatable(axarr[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im_0, cax=cax, orientation='vertical')
        
        im_1 = axarr[1].imshow(pred_i, cmap = 'gray', vmin = pred_i_flatten_mask.min(), vmax = pred_i_flatten_mask.max())
        axarr[1].axis("off")
        axarr[1].set_title("Prediction")
        divider = make_axes_locatable(axarr[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im_1, cax=cax, orientation='vertical')
        mlflow.log_figure(fig, f'dual/time_{i:02d}_{run_name}.png')
        #plt.savefig(path_to_save / f'result_{i}.jpg')
        plt.close(fig)
        
        
        fig = plt.figure()
        single_image = np.stack([ref_i_norm, pred_i_norm, (1-mask)], axis=-1)
        single_image[mask == 0] = [1,1,1]
        plt.imshow(single_image)
        plt.axis("off")
        plt.title(f'Normalized Prediction (Green) and Reference (Red)')
        mlflow.log_figure(fig, f'single/time_{i:02d}_{run_name}.png')
        #plt.savefig(path_to_save / f'single_{i}.jpg')
        plt.close(fig)

        
        #ref_i_norm_flatten_masked = ref_i_norm
        ref_i_flatten_mask = ref_i_norm.flatten()[mask.flatten() == 1]
        pred_i_flatten_mask = pred_i_norm.flatten()[mask.flatten() == 1]
        
        norm_mse_list.append(mean_squared_error(ref_i_flatten_mask, pred_i_flatten_mask))
        norm_mae_list.append(mean_absolute_error(ref_i_flatten_mask, pred_i_flatten_mask))
        
    norm_ref_flatten_mask = rearrange(norm_ref, 'h w c -> (h w) c')[mask.flatten() == 1]
    norm_pred_flatten_mask = rearrange(norm_preds, 'h w c -> (h w) c')[mask.flatten() == 1]
    
    norm_mse = mean_squared_error(norm_ref_flatten_mask, norm_pred_flatten_mask)
    norm_mae = mean_absolute_error(norm_ref_flatten_mask, norm_pred_flatten_mask)
    
    matplotlib.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(14, 6))
    plt.hist([norm_ref_flatten_mask.flatten(), norm_pred_flatten_mask.flatten()], label = ['Reference', 'Prediction'], color = ['red', 'blue'], bins = 80, log=True, rwidth = 0.9, range = [0, 1])
    plt.title('Normalized Prediction and Reference Histograms')
    plt.legend(loc='upper right')
    mlflow.log_figure(fig, f'figures/norm_hist_{run_name}.png')
    plt.close(fig)
    
        
    
    fig = plt.figure(figsize=(12, 5))
    plt.bar(range(len(mse_list)), mse_list)
    plt.ylim([0,0.004])
    plt.ylabel('MSE')
    plt.xticks(range(len(mse_list)))
    plt.xlabel('Time')
    plt.title('Original MSE')
    mlflow.log_figure(fig, f'figures/mse_time_{run_name}.png')
    #plt.savefig(path_to_save / 'mse_time.jpg')
    plt.close(fig)
    
    fig = plt.figure(figsize=(12, 5))
    #plt.plot(mse_list)
    plt.bar(range(len(mae_list)), mae_list)
    plt.ylim([0,0.02])
    plt.ylabel('MAE')
    plt.xticks(range(len(mae_list)))
    plt.xlabel('Time')
    plt.title('Original MAE')
    mlflow.log_figure(fig, f'figures/mae_time_{run_name}.png')
    #plt.savefig(path_to_save / 'mse_time.jpg')
    plt.close(fig)
    
    fig = plt.figure(figsize=(12, 5))
    plt.bar(range(len(norm_mse_list)), norm_mse_list)
    plt.ylim([0,0.004])
    plt.ylabel('MSE')
    plt.xticks(range(len(norm_mse_list)))
    plt.xlabel('Time')
    plt.title('Normalized MSE')
    mlflow.log_figure(fig, f'figures/norm_mse_time_{run_name}.png')
    #plt.savefig(path_to_save / 'mse_time.jpg')
    plt.close(fig)
    
    fig = plt.figure(figsize=(12, 5))
    #plt.plot(mse_list)
    plt.bar(range(len(norm_mae_list)), norm_mae_list)
    plt.ylim([0,0.02])
    plt.ylabel('MAE')
    plt.xticks(range(len(norm_mae_list)))
    plt.xlabel('Time')
    plt.title('Normalized MAE')
    mlflow.log_figure(fig, f'figures/norm_mae_time_{run_name}.png')
    #plt.savefig(path_to_save / 'mse_time.jpg')
    plt.close(fig)
    
    return mse, mae, norm_mse, norm_mae, mse__dict, mae__dict

def generate_images(true_results, predict_results, mask, percentile = None, name = 'comparison'):
    """
    This function generates animated comparison images between true and predicted results for each lag,
    with optional clipping based on percentiles.
    
    @param true_results It looks like the code snippet you provided is a function named
    `generate_images` that generates animated comparison images for true and predicted results. The
    function takes in several parameters including `true_results`, `predict_results`, `mask`, and an
    optional parameter `percentile`.
    @param predict_results It looks like the code snippet you provided is a function named
    `generate_images` that generates animated comparison images for true and predicted results. The
    function takes in several parameters including `true_results`, `predict_results`, `mask`, and an
    optional parameter `percentile`.
    @param mask The `mask` parameter in the `generate_images` function is used to specify a binary mask
    that determines which elements of the `true_i` and `predict_i` arrays should be displayed in the
    generated images. Elements with a value of 1 in the `mask` array will be displayed in
    @param percentile The `percentile` parameter in the `generate_images` function is used to specify
    the percentile value for clipping the true and predicted results. If a `percentile` value is
    provided, the function will clip the true and predicted values based on the specified percentile. If
    `percentile` is set
    """
    matplotlib.rcParams.update({'font.size': 12})
    n_lags = true_results.shape[-1]
    eps = 1e-12
    for lag_i in range(n_lags):
        true_i = true_results[:,:,lag_i]
        predict_i = predict_results[:,:,lag_i]
        
        if percentile is not None:
            true_i_max = np.percentile(true_i, percentile)
            true_i_min = 0 #true_i.min()
            
            predict_i_max = np.percentile(predict_i, percentile)
            predict_i_min = 0 #predict_i.min()
            
            true_i = np.clip(true_i, true_i_min, true_i_max)
            predict_i = np.clip(predict_i, predict_i_min, predict_i_max)
        else:
            true_i_max = true_i.max()
            true_i_min = 0# true_i.min()
            
            predict_i_max = predict_i.max()
            predict_i_min = 0# predict_i.min()
        
        true_i = (true_i - true_i_min + eps) / (true_i_max - true_i_min + eps)
        predict_i = (predict_i - predict_i_min + eps) / (predict_i_max - predict_i_min + eps)
        
        true_i[mask==0] = -1
        predict_i[mask==0] = -1
        
        matplotlib.rcParams.update({'font.size': 18})
        fig = plt.figure(figsize=(12,8))
        plt.axis("off")
        im = plt.imshow(true_i, animated = True)
        
        def animated_func(frame):
            if frame == 0:
                im.set_array(true_i)
                plt.title(f'Reference')
            elif frame == 1:
                im.set_array(predict_i)
                plt.title(f'Prediction')
            return [im]
        
        anim = animation.FuncAnimation(
                                    fig, 
                                    animated_func, 
                                    frames = 2,
                                    interval = 2000, # in ms
                                    )
        with tempfile.TemporaryDirectory() as tmp_dir:
            writer = animation.PillowWriter(fps=1,
                                            metadata=dict(artist='Me'),
                                            bitrate=1800)
            temp_file = Path(tmp_dir) / f'{name}_{lag_i}.gif'
            anim.save(temp_file, writer=writer)
            
            mlflow.log_artifact(temp_file, f'images_({percentile})')
            
        plt.close(fig)

        
def generate_metric_figures(true_results, predict_results, mask, metric, metric_name, run_name, y_limits, bins = [0, 1, 2, 5, 10], log = True):
    """
    The function `generate_metric_figures` generates bar charts for a given metric based on true and
    predicted results, with options for binning and logarithmic scaling.
    
    @param true_results The `true_results` parameter is typically a NumPy array containing the true
    values of the results you are analyzing. It could represent actual data points, ground truth values,
    or reference values that you want to compare against predicted results.
    @param predict_results It seems like you were about to provide more information about the
    `predict_results` parameter, but the message got cut off. Could you please provide more details or
    let me know if you need assistance with something specific related to the `predict_results`
    parameter?
    @param mask The `mask` parameter is used to filter out certain elements from the `true_results` and
    `predict_results` arrays. It is a binary mask that indicates which elements should be included in
    the calculations based on their corresponding positions. Only elements where the mask value is 1
    will be considered in the
    @param metric The `metric` parameter in the `generate_metric_figures` function is a function that
    calculates a performance metric between the true results and predicted results for a specific bin of
    reference values. This metric could be any evaluation metric such as Mean Absolute Error (MAE), Root
    Mean Squared Error (RM
    @param metric_name The `metric_name` parameter in the `generate_metric_figures` function is a string
    that represents the name of the metric being used for evaluation. It could be something like "MAE"
    (Mean Absolute Error), "RMSE" (Root Mean Squared Error), "R2 Score"
    @param run_name The `run_name` parameter is a string that represents the name of the current run or
    experiment. It is used to uniquely identify the figures generated for a specific run in the output
    file names.
    @param y_limits The `y_limits` parameter in the `generate_metric_figures` function is used to
    specify the limits for the y-axis of the plot. It is a tuple containing two values - the lower limit
    and the upper limit for the y-axis. This parameter allows you to control the range of values
    displayed
    @param bins The `bins` parameter in the `generate_metric_figures` function is a list that specifies
    the bin edges for grouping the true results into different ranges. The function then calculates the
    metric between the true and predicted results for each bin.
    @param log The `log` parameter in the `generate_metric_figures` function is a boolean flag that
    determines whether the y-axis of the generated figure should be displayed in logarithmic scale or
    not. If `log` is set to `True`, the y-axis will be displayed in logarithmic scale, otherwise
    
    @return The function `generate_metric_figures` returns two lists: `bins_x` and `bins_y`.
    """
    matplotlib.rcParams.update({'font.size': 14})
    true_results_flatten = rearrange(true_results, 'h w c -> (h w) c')
    predict_results_flatten = rearrange(predict_results, 'h w c -> (h w) c')
    
    true_results_flatten = true_results_flatten[mask.flatten()==1]
    predict_results_flatten = predict_results_flatten[mask.flatten()==1]
    
    bins_x = [f'{bins[0]}']
    true_results_flatten_bin = true_results_flatten[true_results_flatten == bins[0]]
    predict_results_flatten_bin = predict_results_flatten[true_results_flatten == bins[0]]
    bins_y = [metric(true_results_flatten_bin, predict_results_flatten_bin)]
        
    for i, _ in enumerate(bins[:-1]):
        bins_x.append(f'{bins[i]}-{bins[i+1]}')
        true_results_flatten_bin = true_results_flatten[np.logical_and(true_results_flatten > bins[i], true_results_flatten <= bins[i+1])]
        predict_results_flatten_bin = predict_results_flatten[np.logical_and(true_results_flatten > bins[i], true_results_flatten <= bins[i+1])]
        bins_y.append(metric(true_results_flatten_bin, predict_results_flatten_bin))
        
    bins_x.append(f'{bins[-1]}<')
    true_results_flatten_bin = true_results_flatten[true_results_flatten > bins[-1]]
    predict_results_flatten_bin = predict_results_flatten[true_results_flatten > bins[-1]]
    bins_y.append(metric(true_results_flatten_bin, predict_results_flatten_bin))
    
    fig, ax= plt.subplots(figsize = (10, 6))
    #bar = plt.bar(bins_x, bins_y, align='center')
    bar = plt.bar(range(len(bins_x)), bins_y, align='center')
    plt.xticks(range(len(bins_x)), bins_x)
    #plt.xticks(range(len(mae__dict)), list(mae__dict.keys()))
    plt.xlabel('Reference Values Bins (Km2)')
    y_label = f'{metric_name}'
    if log:
        ax.set_yscale('log')
        y_label = f'{y_label} (Log)'
    plt.ylabel(y_label)
    ax.bar_label(bar, fmt='{:,.3f}')
    plt.ylim(y_limits)
    mlflow.log_figure(fig, f'figures/{metric_name}_{run_name}.png')
    plt.close(fig)
    
    return bins_x, bins_y

def generate_histograms(true_results, predict_results, mask, x_limits, run_name, log = True, normalize = False):
    """
    The function `generate_histograms` creates and logs histograms of true and predicted results, with
    options for normalization and logarithmic scaling.
    
    @param true_results True results refer to the actual values or ground truth data that you are
    comparing against. It could be the actual outcomes of a model or experiment.
    @param predict_results Predict_results is a NumPy array containing the predicted results of an
    algorithm for a certain task. It is typically in the format of height x width x channels,
    representing the predicted values for each pixel or region in an image or dataset.
    @param mask The `mask` parameter is likely a binary mask that is used to filter out certain values
    from the `true_results` and `predict_results` arrays. It is used to select specific elements from
    these arrays based on the condition specified by the mask. In this case, it seems to be used to
    @param x_limits The `x_limits` parameter in the `generate_histograms` function is used to specify
    the range of values to be displayed on the x-axis of the histogram. It is a tuple that defines the
    lower and upper limits of the x-axis range for the histogram. For example, if `x_limits
    @param run_name The `run_name` parameter is a string that represents the name of the current run or
    experiment. It is used to uniquely identify the generated histogram figures when logging them for
    tracking purposes.
    @param log The `log` parameter in the `generate_histograms` function is a boolean flag that
    determines whether the y-axis of the histogram should be displayed in logarithmic scale or not. If
    `log` is set to `True`, the y-axis will be displayed in logarithmic scale, otherwise, it
    @param normalize The `normalize` parameter in the `generate_histograms` function is used to specify
    whether the histogram data should be normalized before plotting. When `normalize` is set to `True`,
    the function will normalize the data by scaling it to a range between 0 and 1 based on the standard
    deviation
    """
    matplotlib.rcParams.update({'font.size': 14})
    plt.style.use('seaborn-v0_8-deep')
    
    true_results_flatten = rearrange(true_results, 'h w c -> (h w) c')
    predict_results_flatten = rearrange(predict_results, 'h w c -> (h w) c')
    
    true_results_flatten = true_results_flatten[mask.flatten()==1].flatten()
    predict_results_flatten = predict_results_flatten[mask.flatten()==1].flatten()
    
    if normalize:
        eps = 1e-12
        true_max = 3*true_results_flatten.std()
        true_results_flatten = np.clip(true_results_flatten, 0, true_max)
        true_results_flatten = (true_results_flatten + eps) / (true_results_flatten.max() - true_results_flatten.min() + eps)
        
        pred_max = 3*predict_results_flatten.std()
        predict_results_flatten = np.clip(predict_results_flatten, 0, pred_max)
        predict_results_flatten = (predict_results_flatten + eps) / (predict_results_flatten.max() - predict_results_flatten.min() + eps)
    
    fig = plt.figure(figsize=(14, 6))
    plt.hist([true_results_flatten, predict_results_flatten], 
             label = ['Reference', 'Prediction'], 
             color = ['red', 'blue'], 
             bins = 50, 
             log=log, 
             rwidth = 0.9, 
             range = x_limits)
    if normalize:
        plt.title('Normalized Prediction Histogram')
    else:
        plt.title('Prediction Histogram')
    plt.xlabel('Area (Km2)')
    y_label = 'Count'
    if log:
        y_label = f'{y_label} (Log)'
    plt.ylabel(y_label)
    plt.legend(loc='upper right')
    if normalize:
        mlflow.log_figure(fig, f'figures/hist_{run_name}_norm.png')
    else:
        mlflow.log_figure(fig, f'figures/hist_{run_name}.png')
    plt.close(fig)
    
def evaluate_metric(true_results, predict_results, mask, metric, normalize = False, percentile = 100):
    """
    The function `evaluate_metric` takes true and predicted results, a mask, a metric function, and
    optional parameters to normalize the results and calculates the metric score.
    
    @param true_results True results are the actual values or ground truth data that you are comparing
    the predicted results against.
    @param predict_results The `predict_results` parameter is a multi-dimensional array containing the
    predicted results of a model.
    @param mask The `mask` parameter is a binary mask that is used to select specific elements from the
    `true_results` and `predict_results` arrays. It is used to filter out certain elements based on the
    condition specified in the mask. In this code snippet, `mask.flatten()==1` is used to
    @param metric The `metric` parameter in the `evaluate_metric` function is a function that calculates
    a performance metric to evaluate the true results against the predicted results. This function will
    take the true results and predicted results as inputs and return a numerical value indicating the
    performance of the prediction. Examples of common metrics include Mean
    @param normalize The `normalize` parameter in the `evaluate_metric` function is a boolean flag that
    indicates whether or not to normalize the true and predicted results before evaluating the specified
    metric. If `normalize` is set to `True`, the function will normalize the results based on the
    specified percentile value.
    @param percentile The `percentile` parameter in the `evaluate_metric` function is used to specify
    the percentile value for clipping the true and predicted results when normalization is enabled. This
    parameter determines the upper percentile value to be used for clipping the data.
    
    @return The function `evaluate_metric` returns the result of evaluating the specified metric
    function on the true results and predicted results after processing them based on the provided
    parameters such as mask, normalization, and percentile.
    """
    true_results_flatten = rearrange(true_results, 'h w c -> (h w) c')
    predict_results_flatten = rearrange(predict_results, 'h w c -> (h w) c')
    
    true_results_flatten = true_results_flatten[mask.flatten()==1].flatten()
    predict_results_flatten = predict_results_flatten[mask.flatten()==1].flatten()
    
    if normalize:
        true_max = np.percentile(true_results_flatten, percentile)
        true_min = 0 # true_results_flatten.min()
        true_results_flatten = np.clip(true_results_flatten, true_min, true_max)
        predict_max = np.percentile(predict_results_flatten, percentile)
        predict_min = 0 # predict_results_flatten.min()
        predict_results_flatten = np.clip(predict_results_flatten, predict_min, predict_max)
        
        eps = 1e-12
        true_results_flatten = (true_results_flatten - true_min + eps) / (true_max - true_min + eps)
        predict_results_flatten = (predict_results_flatten - predict_min + eps) / (predict_max - predict_min + eps)
    
    return metric(true_results_flatten, predict_results_flatten)

def integrated_gradients(model, dataloader, device, run_name):
    model = model.to(device)
    model.eval()
    x, y, weight, lag_i, vec_i = next(iter(dataloader))
    y_labels = []
    for k in x.keys():
        for b in range(x[k].shape[1]):
            y_labels.append(f'{k}_{b}')
    sample = model.prepare_input(x)
    ig = IntegratedGradients(model)
    pbar = tqdm(iter(dataloader), desc = 'Evaluating Integrated Gradients')
    attr_results = None
    for data in pbar:
        x, y, weight, lag_i, vec_i = data
        x = model.prepare_input(x).to(device)
        x.requires_grad = True
        model.zero_grad()
        tensor_attributes = ig.attribute(x, ).detach().cpu()
        
        if attr_results is None:
            attr_results = pd.DataFrame(tensor_attributes, columns=y_labels)
        else:
            attr_results = pd.concat([attr_results, pd.DataFrame(tensor_attributes, columns=y_labels)], ignore_index=True)
    sorted_features = attr_results.abs().mean().sort_values().index.to_list()
    
    fig = plt.figure(figsize=(8, 0.5*len(sorted_features)))
    plt.axvline(x = 0, color = 'black', linewidth = 2)
    plt.axvline(x = -8, linestyle = '--', color = 'black')
    plt.axvline(x = -6, linestyle = '--', color = 'black')
    plt.axvline(x = -4, linestyle = '--', color = 'black')
    plt.axvline(x = -2, linestyle = '--', color = 'black')
    plt.axvline(x = 2, linestyle = '--', color = 'black')
    plt.axvline(x = 4, linestyle = '--', color = 'black')
    plt.axvline(x = 6, linestyle = '--', color = 'black')
    plt.axvline(x = 8, linestyle = '--', color = 'black')
    pbar2 = enumerate(tqdm(sorted_features, desc='Plotting data'))
    for i, feature in pbar2:
        scatter_y_labels = [i] * len(attr_results)
        plt.scatter(
            x = attr_results[feature],
            y = scatter_y_labels, 
            c = attr_results[feature],
            cmap = 'jet',
            norm = 'linear',
            vmin = -10,
            vmax = 10
            )
        
    plt.xlim(-10,10)
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xticks(np.linspace(-10, 10, 11))
    plt.suptitle('Integrated Gradients', fontsize=24)
    plt.xlabel('Integrated Gradients Values', fontsize = 18)
    plt.ylabel('Feature', fontsize = 18)
    fig.tight_layout()
    mlflow.log_figure(fig, f'figures/ig_{run_name}.png')
    #plt.savefig('test.png')
    plt.close(fig)
        
def spacially_explicity(true_results, predict_results, mask, kernel):
    pass
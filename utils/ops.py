import json
import numpy as np
import os
import sys
from osgeo import gdal_array
from osgeo import gdal, gdalconst
from pathlib import Path
from typing import Union
import yaml 
from multiprocessing import Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from einops import rearrange
import mlflow

def remove_outliers(img, signficance = 0.01):
    outliers = np.quantile(img, [signficance, 1-signficance], axis = (0,1))
    for channel in range(img.shape[-1]):
        img[:,:,channel] = np.clip(img[:,:,channel], outliers[0, channel],  outliers[1, channel])
    return img


def load_json(fp):
    with open(fp) as f:
        return json.load(f)
    
def save_json(dict_:dict, file_path: Union[str, Path]) -> None:
    """Save a dictionary into a file

    Args:
        dict_ (dict): Dictionary to be saved
        file_path (Union[str, Path]): file path
    """

    with open(file_path, 'w') as f:
        json.dump(dict_, f, indent=4)

def save_yaml(dict_:dict, file_path: Union[str, Path]) -> None:
    """Save a dictionary into a file

    Args:
        dict_ (dict): Dictionary to be saved
        file_path (Union[str, Path]): file path
    """

    with open(file_path, 'w') as f:
        yaml.dump(dict_, f, default_flow_style=False)

def load_yaml(file_path: Union[str, Path]) -> None:
    """Save a dictionary into a file

    Args:
        dict_ (dict): Dictionary to be saved
        file_path (Union[str, Path]): file path
    """

    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
    
def load_opt_image(img_file):
    """load optical data.

    Args:
        img_file (str): path to the geotiff optical file.

    Returns:
        array:numpy array of the image.
    """
    img = gdal_array.LoadFile(str(img_file)).astype(np.float16)
    img[np.isnan(img)] = 0
    if len(img.shape) == 2 :
        img = np.expand_dims(img, axis=0)
    return np.moveaxis(img, 0, -1) / 10000

def load_sb_image(img_file):
    """load a single band geotiff image.

    Args:
        img_file (str): path to the geotiff file.

    Returns:
        array:numpy array of the image. Channels Last.
    """
    img = gdal_array.LoadFile(str(img_file))
    return img

def get_nodata(img_file):
    srs = gdal.Open(str(img_file))
    return srs.GetRasterBand(1).GetNoDataValue()

def load_ml_image(img_file):
    """load a single band geotiff image.

    Args:
        img_file (str): path to the geotiff file.

    Returns:
        array:numpy array of the image. Channels Last.
    """
    img = gdal_array.LoadFile(str(img_file)).astype(np.float32)
    img[np.isnan(img)] = 0
    if len(img.shape) == 2 :
        img = np.expand_dims(img, axis=0)
    return np.moveaxis(img, 0, -1)


def load_SAR_image(img_file):
    """load SAR image, converting from Db to DN.

    Args:
        img_file (str): path to the SAR geotiff file.

    Returns:
        array:numpy array of the image. Channels Last.
    """
    img = gdal_array.LoadFile(str(img_file))
    #img = 10**(img/10) 
    img[np.isnan(img)] = 0
    return np.moveaxis(img, 0, -1)

def save_feature_map(path_to_file, tensor, index = None):
    if index is not None:
        fm = tensor[index]

def create_exps_paths(exp_n):
    exps_path = 'exps'

    exp_path = os.path.join(exps_path, f'exp_{exp_n}')
    models_path = os.path.join(exp_path, 'models')

    results_path = os.path.join(exp_path, 'results')
    predictions_path = os.path.join(results_path, 'predictions')
    visual_path = os.path.join(results_path, 'visual')

    logs_path = os.path.join(exp_path, 'logs')

    
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)

    if not os.path.exists(visual_path):
        os.makedirs(visual_path)

    return exps_path, exp_path, models_path, results_path, predictions_path, visual_path, logs_path

def load_exp(exp_n = None):
    if exp_n is None:
        if len(sys.argv)==1:
            return None
        else:
            return load_json(os.path.join('conf', 'exps', f'exp_{sys.argv[1]}.json'))
    else:
        return load_json(os.path.join('conf', 'exps', f'exp_{exp_n}.json'))
    

def save_geotiff(base_image_path, dest_path, data, dtype, nodata = None):
    """Save data array as geotiff.
    Args:
        base_image_path (str): Path to base geotiff image to recovery the projection parameters
        dest_path (str): Path to geotiff image
        data (array): Array to be used to generate the geotiff
        dtype (str): Data type of the destiny geotiff: If is 'byte' the data is uint8, if is 'float' the data is float32
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

def evaluate_results(reference, predictions, mask, bins = [0, 100]):
    #evaluate metrics
    original_shape = reference.shape[:2]
    ref_flatten_mask = rearrange(reference, 'h w c -> (h w) c')[mask.flatten() == 1]
    pred_flatten_mask = rearrange(predictions, 'h w c -> (h w) c')[mask.flatten() == 1]
    mse = mean_squared_error(ref_flatten_mask, pred_flatten_mask)
    mae = mean_absolute_error(ref_flatten_mask, pred_flatten_mask)
    
    fig = plt.figure()
    plt.hist(ref_flatten_mask.flatten(), bins = 50, log=True, rwidth = 0.9)
    plt.title('Reference Histogram')
    mlflow.log_figure(fig, f'figures/hist_reference.png')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(pred_flatten_mask.flatten(), bins = 50, log=True, rwidth = 0.9)
    plt.title('Prediction Histogram')
    mlflow.log_figure(fig, f'figures/hist_prediction.png')
    plt.close(fig)
    
    
    #metrics with bins
    mse__dict = evaluate_bins(ref_flatten_mask, pred_flatten_mask, bins, mean_squared_error)
    mae__dict = evaluate_bins(ref_flatten_mask, pred_flatten_mask, bins, mean_absolute_error)
    
    
    fig = plt.figure()
    plt.bar(range(len(mse__dict)), list(mse__dict.values()), align='center')
    plt.xticks(range(len(mse__dict)), list(mse__dict.keys()))
    plt.xlabel('bins values (Km2)')
    plt.ylabel('MSE')
    mlflow.log_figure(fig, f'figures/bar_mse.png')
    plt.close(fig)
    
    fig = plt.figure()
    plt.bar(range(len(mae__dict)), list(mae__dict.values()), align='center')
    plt.xticks(range(len(mae__dict)), list(mae__dict.keys()))
    plt.xlabel('bins values (Km2)')
    plt.ylabel('MAE')
    mlflow.log_figure(fig, f'figures/bar_mae.png')
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
        mlflow.log_figure(fig, f'dual/time_{i:02d}.png')
        #plt.savefig(path_to_save / f'result_{i}.jpg')
        plt.close(fig)
        
        
        fig = plt.figure()
        single_image = np.stack([ref_i_norm, pred_i_norm, 0.5*(1-mask)], axis=-1)
        plt.imshow(single_image)
        plt.axis("off")
        mlflow.log_figure(fig, f'single/time_{i:02d}.png')
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
    
    fig = plt.figure()
    plt.hist(norm_ref_flatten_mask.flatten(), bins = 50, log=True, rwidth = 0.9)
    plt.title('Normalized Reference Histogram')
    mlflow.log_figure(fig, f'figures/hist_norm_reference.png')
    plt.close(fig)
    
    fig = plt.figure()
    plt.hist(norm_pred_flatten_mask.flatten(), bins = 50, log=True, rwidth = 0.9)
    plt.title('Normalized Prediction Histogram')
    mlflow.log_figure(fig, f'figures/hist_norm_prediction.png')
    plt.close(fig)
    
    
    
    fig = plt.figure(figsize=(12, 5))
    plt.bar(range(len(mse_list)), mse_list)
    plt.ylim([0,0.004])
    plt.ylabel('MSE')
    plt.xticks(range(len(mse_list)))
    plt.xlabel('Time')
    plt.title('Original MSE')
    mlflow.log_figure(fig, f'figures/mse_time.png')
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
    mlflow.log_figure(fig, f'figures/mae_time.png')
    #plt.savefig(path_to_save / 'mse_time.jpg')
    plt.close(fig)
    
    fig = plt.figure(figsize=(12, 5))
    plt.bar(range(len(norm_mse_list)), norm_mse_list)
    plt.ylim([0,0.004])
    plt.ylabel('MSE')
    plt.xticks(range(len(norm_mse_list)))
    plt.xlabel('Time')
    plt.title('Normalized MSE')
    mlflow.log_figure(fig, f'figures/norm_mse_time.png')
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
    mlflow.log_figure(fig, f'figures/norm_mae_time.png')
    #plt.savefig(path_to_save / 'mse_time.jpg')
    plt.close(fig)
    
    return mse, mae, norm_mse, norm_mae
import geopandas as gpd
import numpy as np
from shapely.geometry import box
import argparse
from pathlib import Path
from rioxarray import open_rasterio
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt
from shutil import rmtree

# Leitura do shapefile
numberOfWindows = 10  # Este valor agora determinará o tamanho da janela de busca
flag_save = True

input_theme_name = "simulado"
sim = "simulado"
real = "real"
range_ = 0  # Definindo o range_ como 0%


reference = open_rasterio(Path('data/validate/reference.tif'))
predictions = open_rasterio(Path('data/validate/transformer_features.tif'))
mask = open_rasterio(Path('data/tiff/mask.tif'))

assert reference.shape == predictions.shape, 'Reference and Predictions must have the same dimensions'

threshold = 0
cell_size = 25000

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
for i in range(1):
    pred_gpd[real] = predictions.values[i].flatten()
    pred_gpd[sim] = reference.values[i].flatten()

pred_gpd = pred_gpd.drop(pred_gpd[pred_gpd['mask'] == 0].index)
pred_gpd['index'] = pred_gpd.index
ref_gpd = ref_gpd.drop(ref_gpd[ref_gpd['mask'] == 0].index)

base_gpd = base_gpd.drop(base_gpd[base_gpd['mask'] == 0].index)

pred_gpd.to_file('data/validate/test.shp')

def MultiRes(gdf, sim, real, numberOfCells, range_):
    col_siml = f'sml{numberOfCells}'  # Similaridade
    col_dfp = f'dfp{numberOfCells}'  # Diferença Percentual
    col_dfa = f'dfa{numberOfCells}'  # Diferença Absoluta
    
    gdf[col_siml] = np.nan
    gdf[col_dfp] = np.nan
    gdf[col_dfa] = np.nan
    
    # Iterar por cada ponto utilizando uma janela retangular de análise
    for idx, cell in tqdm(gdf.iterrows(), total=len(gdf)):
   
        centroid = cell.geometry.centroid
        xmin, ymin = centroid.x, centroid.y
        
        window = box(
            xmin,
            ymin,
            xmin + 25000 * (numberOfCells-1),
            ymin + 25000 * (numberOfCells-1),
        )
        
        neighbors = gdf[gdf.intersects(window)]
        
        diff_abs = neighbors[sim] - neighbors[real]  # Diferenças absolutas (com sinal)
        percent_diffs = np.abs(diff_abs / neighbors[real]) * 100  # Diferenças percentuais
        
        simil_count = np.sum(percent_diffs <= range_)
        simil_count += np.isnan(percent_diffs).sum()
        total_neighbors = len(neighbors)
        similarity_percentage = (simil_count / total_neighbors) * 100
        
        avg_percent_diffs = percent_diffs.mean()
        if np.isnan(avg_percent_diffs):
            avg_percent_diffs = 0
        
        gdf.at[idx, col_siml] = similarity_percentage
        gdf.at[idx, col_dfp] = avg_percent_diffs
        gdf.at[idx, col_dfa] = diff_abs.mean()

        
    accuracy = gdf[col_siml].mean() if len(gdf) > 0 else 0
    return accuracy


MultiRes(pred_gpd, sim, real, 2, range_)
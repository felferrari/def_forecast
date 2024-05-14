import os
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

def create_tiffs_from_shapefile(shapefile_path, compress='lzw', pixel_size=25000):
    # Carrega o shapefile usando geopandas
    gdf = gpd.read_file(shapefile_path)
    
    # Remove as colunas 'id', 'col', 'row', se existirem
    gdf.drop(columns=['id', 'col', 'row'], errors='ignore', inplace=True)
    
    # Cria uma pasta para armazenar os arquivos TIFF, com o mesmo nome do shapefile
    folder_name = os.path.splitext(os.path.basename(shapefile_path))[0]
    os.makedirs(folder_name, exist_ok=True)
    
    # Obtém a extensão total do shapefile
    bounds = gdf.total_bounds
    minx, miny, maxx, maxy = bounds
    
    # Calcula as dimensões do raster
    width = int(np.ceil((maxx - minx) / pixel_size))
    height = int(np.ceil((maxy - miny) / pixel_size))
    
    # Define a transformação do raster
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Para cada atributo, cria um arquivo TIFF
    for attribute in gdf.columns[:-1]:  # Exclui a coluna de geometria
        # Cria um raster vazio
        raster = np.zeros((height, width), dtype=np.float32)
        
        # Rasteriza os valores do atributo para o raster
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[attribute]))
        rasterized = rasterize(shapes=shapes, out=raster, transform=transform, fill=np.nan, all_touched=True, dtype=np.float32)
        
        # Define o caminho do arquivo TIFF de saída
        tiff_path = os.path.join(folder_name, f"{attribute}.tif")
        
        # Cria o arquivo TIFF com compressão LZW
        with rasterio.open(
            tiff_path, 
            'w', 
            driver='GTiff', 
            height=height, 
            width=width, 
            count=1, 
            dtype=str(rasterized.dtype), 
            crs=gdf.crs, 
            transform=transform,
            compress=compress
        ) as dst:
            dst.write(rasterized, 1)
    
    print(f"Arquivos TIFF criados com sucesso na pasta: {folder_name} com compressão {compress.upper()}")

# Substitua o caminho abaixo pelo caminho do seu arquivo Shapefile
shapefile_path = 'data/shp/ArDS_bd_amz_25km.shp'
create_tiffs_from_shapefile(shapefile_path)




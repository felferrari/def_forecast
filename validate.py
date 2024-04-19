import argparse
from pathlib import Path
from rioxarray import open_rasterio
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt
from shutil import rmtree
from shapely.geometry import box

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
        ref_gpd[f'PREV{d:02d}{m:02d}{y:02d}'] = reference.values[i].flatten()
    
    pred_gpd = pred_gpd.drop(pred_gpd[pred_gpd['mask'] == 0].index)
    ref_gpd = ref_gpd.drop(ref_gpd[ref_gpd['mask'] == 0].index)
    pred_gpd.to_file(output_file, layer = 'predictions', driver="GPKG")
    
    base_gpd = base_gpd.drop(base_gpd[base_gpd['mask'] == 0].index)
    #evaluating the multiresolution metrics
    error_abs_cells = []
    error_rel_cells = []
    similarity_cells = []
    for n_cells in range(1, args.max_cells+1):
        similarity_gdp = base_gpd.copy()
        error_abs_gpd = base_gpd.copy()
        error_rel_gpd = base_gpd.copy()
        for idx, cell in tqdm(base_gpd.iterrows(), total = len(base_gpd), desc = f'Evaluating Cell Size {n_cells}', mininterval=0.5):
            #window = cell.geometry.centroid.buffer(cell_size*n_cells+1, cap_style=3)
            centroid = cell.geometry.centroid
            xmin, ymin = centroid.x, centroid.y
            
            window = box(
                xmin,
                ymin,
                xmin + 25000 * (n_cells-1),
                ymin + 25000 * (n_cells-1),
            )
            
            pred_neighbors = pred_gpd[pred_gpd.geometry.intersects(window)]
            ref_neighbors = ref_gpd[ref_gpd.geometry.intersects(window)]
            
            for biweek in biweeks:
                error_abs = (pred_neighbors[biweek] - ref_neighbors[biweek]).abs()  # Diferenças absolutas (com sinal)
                error_rel = np.abs(error_abs / ref_neighbors[biweek]) * 100  # Diferenças percentuais
                
                simil_count = np.sum(error_rel <= threshold)
                simil_count += np.isnan(error_rel).sum()
                total_neighbors = len(pred_neighbors)
                similarity_percentage = (simil_count / total_neighbors) * 100
                
                avg_percent_diffs = error_rel.mean()
                if np.isnan(avg_percent_diffs):
                    avg_percent_diffs = 0
                
                error_rel_gpd.at[idx, biweek] = error_abs.mean()
                error_abs_gpd.at[idx, biweek] = avg_percent_diffs
                similarity_gdp.at[idx, biweek] = similarity_percentage
        
        error_abs_gpd.to_file(output_file, layer = f'avg_diff_abs_perc_{n_cells}', driver="GPKG")
        error_rel_gpd.to_file(output_file, layer = f'avg_diff_abs_{n_cells}', driver="GPKG")
        similarity_gdp.to_file(output_file, layer = f'similarity_{n_cells}', driver="GPKG")
        
        error_abs_gpd.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        error_abs_biweeks = []
        error_rel_biweeks = []
        similarity_biweeks = []
        for biweek in tqdm(biweeks, desc=f'Generating figures for Cell size {n_cells}', mininterval=0.5):
            error_abs_biweeks.append(error_rel_gpd[biweek].mean())
            error_rel_biweeks.append(error_abs_gpd[biweek].mean())
            similarity_biweeks.append(similarity_gdp[biweek].mean())
            
            fig, ax = plt.subplots(1,1,figsize=(12,8))
            error_rel_gpd.plot(ax = ax, column=biweek, legend=True, legend_kwds={"label": "Absolute Error"})
            plt.axis("off")
            plt.title(f'Absolute Error ({biweek[4:]})')
            plt.savefig(output_figures/f'diff_abs_{n_cells}_{biweek[4:]}.png')
            plt.close(fig)
            
            fig, ax = plt.subplots(1,1,figsize=(12,8))
            error_abs_gpd.plot(ax = ax, column=biweek, legend=True, legend_kwds={"label": "Percentual Error"}, missing_kwds={"color": "lightgrey"}, vmin = 0, vmax = 100)
            plt.axis("off")
            plt.title(f'Percentual Error ({biweek[4:]})')
            plt.savefig(output_figures/f'diff_rel_{n_cells}_{biweek[4:]}.png')
            plt.close(fig)
            
            fig, ax = plt.subplots(1,1,figsize=(12,8))
            similarity_gdp.plot(ax = ax, column=biweek, legend=True, legend_kwds={"label": "Similarity"})
            plt.axis("off")
            plt.title(f'Similarity ({biweek[4:]})  (threshold = {threshold}%)')
            plt.savefig(output_figures/f'similarity_{n_cells}_{biweek[4:]}.png')
            plt.close(fig)
            
        fig, ax = plt.subplots(1,1,figsize=(16,5))
        plt.bar(dates, error_abs_biweeks)
        plt.title(f'Absolute Error')
        plt.ylabel('Average Absolute Error')
        plt.xlabel('Biweek')
        plt.xticks(rotation=315)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='left')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(14)
        fig.tight_layout()
        plt.savefig(output_figures/f'time_diff_abs_{n_cells}.png')
        plt.close(fig)
        
        fig, ax = plt.subplots(1,1,figsize=(16,5))
        plt.bar(dates, error_rel_biweeks)
        plt.title(f'Percentual Error')
        plt.ylabel('Average Percentual Error (Log)')
        plt.yscale('log')
        plt.xlabel('Biweek')
        plt.xticks(rotation=315)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='left')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(14)
        fig.tight_layout()
        plt.savefig(output_figures/f'time_diff_rel_{n_cells}.png')
        plt.close(fig)
        
        fig, ax = plt.subplots(1,1,figsize=(16,5))
        plt.bar(dates, error_abs_biweeks)
        plt.title(f'Similarity (threshold = {threshold}%)')
        plt.ylabel('Average Similarity')
        plt.xlabel('Biweek')
        plt.xticks(rotation=315)
        plt.setp(ax.xaxis.get_majorticklabels(), ha='left')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(14)
        fig.tight_layout()
        plt.savefig(output_figures/f'time_similarity_{n_cells}.png')
        plt.close(fig)
        
        error_abs_cells.append(np.array(error_abs_biweeks).mean())
        error_rel_cells.append(np.array(error_rel_biweeks).mean())
        similarity_cells.append(np.array(similarity_biweeks).mean())
        
    fig = plt.figure(figsize=(12,8))
    #plt.bar(range(args.max_cells+1), average_diff_abs_cells)
    plt.plot(range(1, args.max_cells+1), error_abs_cells)
    plt.title(f'Absolute Error')
    plt.ylabel('Average Absolute Error')
    plt.xlabel('Cells Size')
    plt.savefig(output_figures/f'size_diff_abs.png')
    plt.close(fig)
    
    fig = plt.figure(figsize=(12,8))
    plt.plot(range(1, args.max_cells+1), error_rel_cells)
    plt.title(f'Percentual Error')
    plt.ylabel(f'Average Percentual Error')
    plt.xlabel('Cells Size')
    plt.savefig(output_figures/f'size_diff_rel.png')
    plt.close(fig)
    
    fig = plt.figure(figsize=(12,8))
    plt.plot(range(1, args.max_cells+1), similarity_cells)
    plt.title(f'Similarity (threshold = {threshold}%)')
    plt.ylabel(f'Average Similarity')
    plt.xlabel('Cells Size')
    plt.savefig(output_figures/f'size_similarity.png')
    plt.close(fig)
    
if __name__ == '__main__':
    main()
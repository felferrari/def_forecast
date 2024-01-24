from utils.ops import load_sb_image
from pathlib import Path

path = Path(r'D:\Ferrari\Projects\deforestation-forescast\data\INPE')

a = load_sb_image(path / 'amz_25km_apa.tif')


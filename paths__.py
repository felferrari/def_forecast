from pathlib import Path


#PATHS
base_data_path = Path(r'/home/felferrari/projects/def_forecast/data')
path_to_mask = base_data_path / r'tiff/mask.tif'
path_to_data = {
    'mask': base_data_path / r'tiff/mask.tif',
    'def_data':  base_data_path / r'tiff/ArCS.tif',
    'quinz' : {
        'DeAr': base_data_path / r'tiff/DeAr.tif',
        'Monthly': base_data_path / r'tiff/monthly.tif',
        'Forest': base_data_path / r'tiff/flor.tif',
        'Cloud': base_data_path / r'tiff/nv.tif',
    },
    'atemporal' : {
        'DistPorts' : base_data_path / r'tiff/distport.tif',
        'DistRios' : base_data_path / r'tiff/distrios.tif',
        'DistUrbs' : base_data_path / r'tiff/distUrb.tif',
        'EFAMS_APA' : base_data_path / r'tiff/EFAMS_APA.tif',
        'EFAMS_ASS' : base_data_path / r'tiff/EFAMS_ASS.tif',
        'EFAMS_CAR' : base_data_path / r'tiff/EFAMS_CAR.tif',
        'EFAMS_FPND' : base_data_path / r'tiff/EFAMS_FPND.tif',
        'EFAMS_IND' : base_data_path / r'tiff/EFAMS_IND.tif',
        'EFAMS_TI' : base_data_path / r'tiff/EFAMS_TI.tif',
        'EFAMS_UC' : base_data_path / r'tiff/EFAMS_UC.tif',
        'HIDR' : base_data_path / r'tiff/hidr.tif',
        'NOFOREST' : base_data_path / r'tiff/nf.tif',
        'RodNOficial' : base_data_path / r'tiff/rodnofic.tif',
        'RodOficial' : base_data_path / r'tiff/rodofic.tif',
    }
}
import zipfile

with zipfile.ZipFile('features.tar', 'r') as zip_ref:
    zip_ref.extractall('data/dusha')

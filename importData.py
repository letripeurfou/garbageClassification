
import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'garbage-classification:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F81794%2F189983%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240129%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240129T104010Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D523db9adfe19eca4a7f1b2005fd9e5f44256d146fac204d1d5125a920c3d33f6c13904e08660c393c7ee9605fbfc4248bfa4d98b00d557f0f435d5896b07bb0c8782a3012fac09ecda3e0a37f068230f9a6f2e016043b7583aeadcc564c9bdcc59d337f1286bf82553edc636b0e2ebf689fc53d2c80b4a61cd87427fa064192dc0eff79a576d22dd46d56a17bc4c8c2ad236dde72f16ebca603264ec123491c86cdff3acc07f09e1aafb9a7adff2efc9bc5d086249575c7fdbecb9c501ff4a5a215d73f7fd3be64ce53055467192696efc04b8c29ec708bba58bce4f80695297939af70afb702b18630420c1c3fb9fa40eb1445bca533dda3ff16def77a8326f,d/mostafaabla/garbage-classification:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F1115942%2F1874598%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240129%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240129T104010Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D1c053feab995997b9497525d651b6dd5ff45f00fea28abb396a4e856f32d604574ea99148d394b17935770330423d7fc6f21dc6b668a6a8ef3498765f6b964e62d8cab1a2199722458c562fc2b51ac69d1dc06e92e395b460ae4647542aa1583b8c191710e61d93340fddc80e5a7b13f04e7c05a661614253fcd1e97429bc25cef1ffc113d541d08cc893415b41f6d3f9a25356c97c0db6012f10e24cfeec76bc161b14436b6536639a285e8beb6de9e786c6bef29531a708cf4f45b1a7563cbae2aaddfe24bd29d1014ecfd40fb8bbd80fb1ea9581e0491d46bd80423a1d352a710af297a786551401a5d9f29581506c8b4c48984a9c84d1d4e65bdef949e31'

KAGGLE_INPUT_PATH='./data/garbage-classification'
KAGGLE_WORKING_PATH='./data/working'
#KAGGLE_SYMLINK='kaggle'

shutil.rmtree('./data/garbage-classification', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

# try:
#   os.symlink(KAGGLE_INPUT_PATH, os.path.join(".", 'garbage-classification'), target_is_directory=True)
# except FileExistsError:
#   pass
# try:
#   os.symlink(KAGGLE_WORKING_PATH, os.path.join(".", 'working'), target_is_directory=True)
# except FileExistsError:
#   pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

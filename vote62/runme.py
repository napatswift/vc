import os
import subprocess
import pandas
import random

if __name__ == '__main__':
    metadata = dict(image=[], province=[], district=[], subdistrict=[], electoral_district_number=[])
    image_dir = os.path.join('data', 'image')
    os.makedirs(image_dir, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk('กรุงเทพมหานคร'):
        image_names = [fn for fn in filenames if fn.endswith('.jpeg') or fn.endswith('.jpg') or fn.endswith('.png')]
        if not image_names or len(dirpath.split('/')) != 5: continue
        prov, distr, subdistr, electoral_num, _ = dirpath.split('/')
        for file in image_names:
            metadata['image'].append(os.path.join(image_dir, file))
            metadata['province'].append(prov)
            metadata['district'].append(distr)
            metadata['subdistrict'].append(subdistr)
            metadata['electoral_district_number'].append(electoral_num)
            subprocess.call(['cp', os.path.join(dirpath, file), image_dir])

    pandas.DataFrame(metadata).to_csv(os.path.join('data','metadata.txt'), index=0, sep='\t')
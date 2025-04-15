class rep_imgs:

    def __init__(self, path, *, ext='png'):

        self.ext=ext

        if path.endswith('.zip'):
            import zipfile
            self.ar=zipfile.ZipFile(path, mode='r')
            self.fnames=self.ar.namelist()

            import pathlib

            self.fnames=[e for e in self.fnames if pathlib.PurePath(e).match(f'*.{ext}')]

            #print(f'fnames {self.fnames[:10]}')
        else:
            import glob    
            self.fnames=glob.glob(f'{path}/*.{ext}')
            self.ar=None

        self.fnames=sorted(self.fnames)
        
    def __getitem__(self, key):
        fname=self.fnames[key]
        assert fname.endswith(self.ext)

        import matplotlib.pyplot as plt

        if self.ar is None:
            im = plt.imread(fname)
        else:

            import io
            import numpy as np
            
            data=self.ar.read(fname)
            dataEnc = io.BytesIO(data)
            
            im = plt.imread(dataEnc)            
        
        return im

    def __len__(self):
        return len(self.fnames)

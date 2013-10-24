# This code is supporting material for the book
# Building Machine Learning Systems with Python
# by Willi Richert and Luis Pedro Coelho
# published by PACKT Publishing
#
# It is made available under the MIT License

#import milksets.iris
#import milksets.seeds
import urllib2,os


def save_as_tsv(fname, module):
    features, labels = module.load()
    nlabels = [module.label_names[ell] for ell in labels]
    with open(fname, 'w') as ofile:
        for f, n in zip(features, nlabels):
            print >>ofile, "\t".join(map(str, f) + [n])

#save_as_tsv('../data/iris.tsv', milksets.iris)
#save_as_tsv('../data/seeds.tsv', milksets.seeds)
def download_file(url,file_name=None):    
    if not file_name: file_name = url.split('/')[-1]
    u = urllib2.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file_name, file_size)
    os.system('cls')
    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,

    f.close()
    return file_name

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
filename = download_file(url,"../../data/seeds.tsv")
print(filename)

from sklearn.datasets import load_iris
import numpy as np

data = load_iris()
features = data['data']
labels = data['target_names'][data['target']]
filename = "../../data/iris.tsv"
with open(filename, 'w') as ofile:
    for f, n in zip(features, labels):
        print >>ofile, "\t".join(map(str, f) + [n])
print(filename)
# Data Set Information:
# 
# The examined group comprised kernels belonging to three different varieties of wheat: Kama, Rosa and Canadian, 70 elements each, randomly selected for 
# the experiment. High quality visualization of the internal kernel structure was detected using a soft X-ray technique. It is non-destructive and considerably cheaper than other more sophisticated imaging techniques like scanning microscopy or laser technology. The images were recorded on 13x18 cm X-ray KODAK plates. Studies were conducted using combine harvested wheat grain originating from experimental fields, explored at the Institute of Agrophysics of the Polish Academy of Sciences in Lublin. 
# 
# The data set can be used for the tasks of classification and cluster analysis.
#see http://archive.ics.uci.edu/ml/datasets/seeds for column names

# To construct the data, seven geometric parameters of wheat kernels were measured: 
# 1. area A, 
# 2. perimeter P, 
# 3. compactness C = 4*pi*A/P^2, 
# 4. length of kernel, 
# 5. width of kernel, 
# 6. asymmetry coefficient 
# 7. length of kernel groove. 
# All of these parameters were real-valued continuous.
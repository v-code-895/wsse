"""
Download the third-party data used by the experiments into data/ (see DATA.md for sources and
licences; none of it is redistributed in this repository).

  python scripts/download_data.py mobilenet        MobileNetV2 ImageNet weights + class index
  python scripts/download_data.py afhq             AFHQ animal faces, Kaggle mirror
  python scripts/download_data.py all

Kaggle downloads may require a Kaggle account: if the direct download is refused, download the
archive from the dataset page by hand and place it at the path printed below, then re-run.
"""
import os, sys, urllib.request, zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
SOURCES = {
    "mobilenet": [("mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5",
                   "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5", False),
                  ("mobilenet_v2/imagenet_class_index.json",
                   "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json", False)],
    "afhq": [("afhq/archive.zip", "https://www.kaggle.com/api/v1/datasets/download/andrewmvd/animal-faces", True)],
}

def fetch(rel, url, unzip):
    path = os.path.join(ROOT, rel); os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"downloading {url}\n         -> {os.path.normpath(path)}")
        try: urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"   failed ({e}); download it by hand to the path above and re-run"); return
    if unzip:
        out = os.path.join(os.path.dirname(path), "data")
        if not os.path.exists(out):
            with zipfile.ZipFile(path) as z: z.extractall(out)
            print(f"   extracted to {os.path.normpath(out)}")

if __name__ == "__main__":
    what = sys.argv[1:] or ["all"]
    for k in (SOURCES if "all" in what else what):
        for rel, url, unzip in SOURCES[k]: fetch(rel, url, unzip)

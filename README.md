# Setup
### Data
First create ./data/ and ./data_unpacked/ directories

Make sure you have these files placed in ./data/ directory:
1. `fddb_annotations.tgz`
2. `fddb_images.tar.gz`
3. `shape_predictor_68_face_landmarks.dat.bz2`

After that execute these three commands:
1. `tar -xf data/fddb_annotations.tgz -C data_unpacked/`
2. `tar -xf data/fddb_images.tar.gz -C data_unpacked/`
3. `bzip2 -d data/shape_predictor_68_face_landmarks.dat.bz2 > data_unpacked/shape_predictor_68_face_landmarks.dat`

### Python packages
Execute following command:
- `pip3 install -r requirements.txt`

# Running face detection pipeline:
- `python3 src/main.py --num-pictures-to-process N`

where `N` is number of pictures from which to extract faces

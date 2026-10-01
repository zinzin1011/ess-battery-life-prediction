# 데이터

원본 데이터는 용량이 커서(배치당 2~3 GB) 저장소에 포함하지 않습니다.

1. Kaggle(또는 https://data.matr.io/1/)에서 아래 3개 파일을 내려받습니다.
   - `2017-05-12_batchdata_updated_struct_errorcorrect.mat` (Batch 1, 학습)
   - `2018-02-20_batchdata_updated_struct_errorcorrect.mat` (Batch 2, 평가)
   - `2018-04-12_batchdata_updated_struct_errorcorrect.mat` (Batch 3, 추가 평가)
2. `data/raw/` 폴더에 넣거나, 실행 시 `--data-dir`로 위치를 지정합니다.

파일은 MATLAB v7.3(HDF5) 형식이라 `scipy.io.loadmat`이 아니라 `h5py`로 읽습니다.

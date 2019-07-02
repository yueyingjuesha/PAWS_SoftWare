import make_data_pandas
from mydataset import DataSet
import os


def main_predict(basepath, method='xgb',
                 column_names=None, files=None, selected_features=None, patrol=None, poaching=None):
  ##########################################################################
  ##########################################################################
  ##########################################################################
  ##########################################################################
  check_file = [
      'X', 'Y', 'is-toy_patrol', 'is-toy_poaching', 'is-toy_road',
      'dist-toy_patrol', 'dist-toy_poaching', 'dist-toy_road',
      'toy_altitude', 'is-toy_patrol', 'is-toy_poaching']

  for i in check_file:
    if i+'.csv' not in os.listdir(basepath):
      return (False,i)
  
  column_names = [
      'X',
      'Y',
      'is-toy_patrol',
      'is-toy_poaching',
      'is-toy_road',
      'dist-toy_patrol',
      'dist-toy_poaching',
      'dist-toy_road',
      'toy_altitude',
  ]
  files = []
  for name in column_names:
    files.append(f"{basepath}/{name}.csv")
  selected_features = [
      "is-toy_road",
      "normal-dist-toy_road",
      "normal-toy_altitude",
  ]
  patrol = 'is-toy_patrol'
  poaching = 'is-toy_poaching'
  ##########################################################################
  ##########################################################################
  ##########################################################################
  ##########################################################################
  # final_data: originally: final.csv
  final_data = make_data_pandas.process_automate_data(files, column_names)

  ##########################################################################

  df_alldata, df_invaliddata, df_unknowndata, df_allpositive, \
      df_allnegative, df_slct_positive, df_slct_negative, \
      df_slct_unlabeled, \
      PositiveData, NegativeData, UnknownData = \
      make_data_pandas.preprocessing_fn1(
          final_data, patrol, poaching, selected_features)

  df_alldata2, df_validdata2, df_invaliddata2, df_slct_valid, NewAllData = \
      make_data_pandas.preprocessing_fn2(final_data, selected_features)

  neg, NegativeData, NotFam, neg_label, Fam, dataset = \
      make_data_pandas.build_dataset(
          PositiveData, NegativeData, UnknownData, FoldNum=4)

  # name of text file output for probabilistic predictions of
  # each grid cell in conservations 1 and 2
  qgis_file_in1 = "predictions1.txt"
  qgis_file_in2 = "predictions2.txt"
  qgis_file_in1_str = make_data_pandas.main_poaching_predict(
      qgis_file_in1,
      qgis_file_in2,
      df_allpositive,
      df_allnegative,
      df_unknowndata,
      df_validdata2,
      df_slct_positive,
      df_slct_negative,
      df_slct_unlabeled,
      df_slct_valid,
      dataset,
      df_invaliddata,
      df_invaliddata2,
      method,
  )
  return (qgis_file_in1_str, df_alldata)


def main_prep_qgis(output,
                   qgis_file_out1="predictions_heatmap1.asc"):
  # qgis_file_out1: raster file of probabilistic predictions

  # represents the coordinates of the left bottom corner for
  # conservation site 1 (longitude and latitude if working with WGS84)
  qgis_file_in1_str, df_alldata = output
  xcorner1 = 127.76402335
  ycorner1 = 43.5257568717

  # define the grid sizes (discretization levels) for each conservations ite
  # which should match from the automate_data.py script
  gridDim1 = 0.01
  make_data_pandas.prep_qgis(qgis_file_in1_str, qgis_file_out1, gridDim1,
                             xcorner1, ycorner1, df_alldata)
  return




def main_toy():
  basepath = '/Users/hukai/Downloads/PAWS_SoftWare-master/Data/csv_output/'
  method = 'dt' #'xgb', 'svm', 'dt'
  save = 'pred_toy.asc'
  output = main_predict(basepath, method)
  main_prep_qgis(output, save)
  print('done')

if __name__ == '__main__':
  main_toy()

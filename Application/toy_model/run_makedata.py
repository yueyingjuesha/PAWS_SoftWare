import make_data_pandas
from mydataset import DataSet
import os

def run_makedata(basepath):

  files = [
      f'{basepath}/X.csv',
      f'{basepath}/Y.csv',
      f'{basepath}/is-toy_patrol.csv',
      f'{basepath}/is-toy_poaching.csv',
      f'{basepath}/is-toy_road.csv',
      f'{basepath}/dist-toy_patrol.csv',
      f'{basepath}/dist-toy_poaching.csv',
      f'{basepath}/dist-toy_road.csv',
      f'{basepath}/toy_altitude.csv',
  ]

  contents = os.listdir(basepath)
  for file in files:
    if file not in files:
      return file

  column_names = ['X', 'Y',
                  'is-toy_patrol', 'is-toy_poaching', 'is-toy_road',
                  'dist-toy_patrol', 'dist-toy_poaching', 'dist-toy_road',
                  'toy_altitude']

  make_data_pandas.process_automate_data(files, column_names)

  ##########################################################################

  fn1 = "final.csv"
  fn2 = "final.csv"
  # name of text file output for probabilistic predictions of
  # each grid cell in conservations 1 and 2
  qgis_file_in1 = "predictions1.txt"
  qgis_file_in2 = "predictions2.txt"
  # raster file of probabilistic predictions
  qgis_file_out1 = "predictions_heatmap1.asc"
  qgis_file_out2 = "predictions_heatmap2.asc"
  # specify which features to use from final.csv feature spreadsheet
  selected_features = [
      "is-toy_road",
      "normal-dist-toy_road",
      "normal-toy_altitude",
  ]
  # specify which feature symbolizes where patrolling occurs
  patrol = 'is-toy_patrol'
  # specify which feature symbolizes where poaching occurs
  poaching = 'is-toy_poaching'

  ##########################################################################

  df_alldata, df_invaliddata, df_unknowndata, df_allpositive, \
      df_allnegative, df_slct_positive, df_slct_negative, \
      df_slct_unlabeled, \
      PositiveData, NegativeData, UnknownData = \
      make_data_pandas.preprocessing_fn1(
          fn1, patrol, poaching, selected_features)

  df_alldata2, df_validdata2, df_invaliddata2, df_slct_valid, NewAllData = \
      make_data_pandas.preprocessing_fn2(fn2, selected_features)

  FoldNum = 4
  neg, NegativeData, NotFam, neg_label, Fam, dataset = \
      make_data_pandas.build_dataset(
          PositiveData, NegativeData, UnknownData, FoldNum=FoldNum)

  ##########################################################################

  make_data_pandas.main_poaching_predict(qgis_file_in1, qgis_file_in2,
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
                                         df_invaliddata2
                                         )

  # represents the coordinates of the left bottom corner for
  # conservation site 1 (longitude and latitude if working with WGS84)
  xcorner1 = 127.76402335
  ycorner1 = 43.5257568717

  # define the grid sizes (discretization levels) for each conservations ite
  # which should match from the automate_data.py script
  gridDim1 = 0.01
  make_data_pandas.prep_qgis(qgis_file_in1, qgis_file_out1, gridDim1,
                             xcorner1, ycorner1, df_alldata)
  return 'Finished!'

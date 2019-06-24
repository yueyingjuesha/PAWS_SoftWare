import numpy as np
import pandas as pd
import sys
#from sklearn import tree
#from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import roc_auc_score
from mydataset import DataSet
import xgboost as xgb


def process_automate_data(files, column_names):
  '''Output final.csv for future use.
  The output file not only includes all the column names mentoned above,
  but also the normalized ones.'''
  raw_df_list = []
  # read files into dataframe
  for f in files:
    print("process_automate_data| file: ", f)
    # a list of <class 'pandas.core.frame.DataFrame'>
    raw_df_list.append(pd.read_csv(f))

  # get the DN as a dataframe
  # <class 'pandas.core.frame.DataFrame'>
  DN_df = raw_df_list[0][['DN']].sort_values(by=['DN'])
  DN_df.reset_index(inplace=True)

  # rename columns and sort based on DN
  select_df_list = []
  for i in range(0, len(raw_df_list)):
    # rename columns
    # <class 'pandas.core.indexes.base.Index'>
    raw_df_list[i].columns = ['DN', column_names[i]]

    # sort by DN
    # <class 'pandas.core.frame.DataFrame'>
    cur_sorted_df = raw_df_list[i].sort_values(by=['DN'])
    cur_sorted_df.reset_index(inplace=True)

    # select revelant columns
    cur_select_df = cur_sorted_df[[column_names[i]]]

    # normalize the selected columns
    cur_normalized_df = (cur_select_df - cur_select_df.min()) / \
        (cur_select_df.max() - cur_select_df.min())
    cur_normalized_df.columns = ["normal-" + column_names[i]]

    select_df_list.append(cur_select_df)
    if column_names[i][0:3] != 'is-':
      select_df_list.append(cur_normalized_df)

  # concatenate columns
  select_df_list = [DN_df] + select_df_list
  # <class 'pandas.core.frame.DataFrame'>
  comb_DN_ABC = pd.concat(select_df_list, axis=1)
  comb_DN_ABC.sort_values(by=["DN"], inplace=True)
  # TODO: this drop does not work. If intended, please add: inplace=True
  comb_DN_ABC.drop(['index'], axis=1)
  comb_DN_ABC.to_csv("final.csv")


def preprocessing_fn1(fn1, patrol, poaching, selected_features):

  print("====start preprocessing_fn1====")

  df_alldata = pd.read_csv(fn1)
  # select data without nan, used only here
  df_validdata = df_alldata.dropna()
  # select data with nan
  df_invaliddata = df_alldata[df_alldata.isnull().any(axis=1)]
  # select labeled data, used only here
  df_knowndata = df_validdata[(df_validdata[patrol] > 0)]
  # select unlabeled data
  df_unknowndata = df_validdata[(df_validdata[patrol] == 0)]
  # obtain positive data, replace 'Poaching-17' and others with feature
  # names that specify existence of previous poaching
  df_allpositive = df_knowndata[(df_knowndata[poaching] != 0)]

  df_allnegative = df_knowndata[(df_knowndata[poaching] == 0)]


  df_slct_positive = df_allpositive[selected_features]
  df_slct_negative = df_allnegative[selected_features]
  df_slct_unlabeled = df_unknowndata[selected_features]
  # <class 'numpy.ndarray'>
  PositiveData = df_slct_positive.values
  NegativeData = df_slct_negative.values
  UnknownData = df_slct_unlabeled.values
  print(f"PositiveData #: {len(PositiveData)}")
  print(f"NegativeData #: {len(NegativeData)}")
  print(f"UnknownData  #: {len(UnknownData)}")

  print("====done preprocessing_fn1====")

  return df_alldata, df_invaliddata, df_unknowndata, df_allpositive, \
      df_allnegative, df_slct_positive, df_slct_negative, \
      df_slct_unlabeled, \
      PositiveData, NegativeData, UnknownData

# df_alldata, df_invaliddata, df_unknowndata, df_allpositive, \
#     df_allnegative, df_slct_positive, df_slct_negative, \
#     df_slct_unlabeled, \
#     PositiveData, NegativeData, UnknownData = preprocessing_fn1(fn1)

##########################################################################


def preprocessing_fn2(fn2, selected_features):

  df_alldata2 = pd.read_csv(fn2)
  # select data without nan
  df_validdata2 = df_alldata2.dropna()
  # select data with nan
  df_invaliddata2 = df_alldata2[df_alldata2.isnull().any(axis=1)]
  df_slct_valid = df_alldata2[selected_features]
  NewAllData = df_slct_valid.values
  # used: df_validdata2, df_invaliddata2, df_slct_valid
  return df_alldata2, df_validdata2, df_invaliddata2, df_slct_valid, NewAllData




def build_dataset(PositiveData, NegativeData, UnknownData, FoldNum=4):
  '''fill in Dataset class for future use'''

  # not used
  fold_pos_num = len(PositiveData) // FoldNum
  fold_neg_num = len(NegativeData) // FoldNum

  # shuffle the negative data
  np.random.shuffle(NegativeData)
  neg = NegativeData[:fold_neg_num]
  NegativeData = NegativeData[fold_neg_num:]

  # negative sampling here
  sample_size = NegativeData.shape[0]
  indx = np.random.randint(UnknownData.shape[0], size=sample_size)
  Udata = UnknownData[indx]

  # <class 'numpy.ndarray'>
  # We add more negative data by sampling from UnknownData
  NotFam = np.concatenate((Udata, NegativeData), axis=0)
  neg_label = np.array([0.] * len(neg))
  Fam = PositiveData

  dataset = DataSet(positive=Fam, negative=NotFam, fold_num=FoldNum)

  # Except for dataset, all others are not used
  return neg, NegativeData, NotFam, neg_label, Fam, dataset




def main_poaching_predict(qgis_file_in1, qgis_file_in2,
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
                          ):
  ''' Generate the actual predictions by training on all the data
  qgis_file_in{1,2} is the output of this function,
  which contains the probabilistic predictions as a text file.'''
  print("====start main_poaching_predict====")

  #
  train_data, train_label = dataset.get_train_all_up(100)
  param = {
      'max_depth': 10,
      'eta': 0.1,
      'silent': 1,
      'objective': 'binary:logistic'
  }
  num_round = 1000
  D_train = xgb.DMatrix(train_data, label=train_label)
  # <class 'xgboost.core.Booster'>
  bst = xgb.train(param, D_train, num_round)

  #
  PositiveDataID = df_allpositive["DN"].values
  NegativeDataID = df_allnegative["DN"].values
  UnknownDataID = df_unknowndata["DN"].values

  ALLID = list(PositiveDataID) + list(NegativeDataID) + list(UnknownDataID)
  ALLDATA = list(df_slct_positive.values) + \
      list(df_slct_negative.values) + \
      list(df_slct_unlabeled.values)

  NEWALLID = list(df_validdata2["DN"].values)
  NEWALLDATA = list(df_slct_valid.values)

  ##########################################################################

  ALLDATA = np.array(ALLDATA)

  D_ALLDATA = xgb.DMatrix(ALLDATA)

  # prediction results
  ALL_value = bst.predict(D_ALLDATA)
  ALL_scores = np.zeros(len(ALL_value))

  for i in range(0, len(ALL_value)):
    if (ALL_value[i] > 0.5):
      ALL_scores[i] = 1.0
    else:
      ALL_scores[i] = 0.0

  id_label = zip(ALLID, ALL_value)
  id_label = list(id_label)

  Invalid_ID = df_invaliddata["DN"].values
  for id in Invalid_ID:
    id_label.append((id, 0.0))

  id_label = sorted(id_label, key=lambda x: x[0], reverse=False)
  with open(qgis_file_in1, 'w') as fout:
    fout.write('ID\tLabel\n')
    for idx, label in id_label:
      temp_str = str(idx) + '\t' + str(label) + '\n'
      fout.write(temp_str)

  

  NEWALLDATA = np.array(NEWALLDATA)

  D_NEWALLDATA = xgb.DMatrix(NEWALLDATA)

  # prediction results
  ALL_newvalue = bst.predict(D_NEWALLDATA)
  ALL_newscores = np.zeros(len(ALL_newvalue))

  for i in range(0, len(ALL_newvalue)):
    if (ALL_newvalue[i] > 0.5):
      ALL_newscores[i] = 1.0
    else:
      ALL_newscores[i] = 0.0

  newid_label = zip(NEWALLID, ALL_newvalue)
  newid_label = list(newid_label)

  Invalid_ID = df_invaliddata2["DN"].values
  for id in Invalid_ID:
    newid_label.append((id, 0.0))

  newid_label = sorted(newid_label, key=lambda x: x[0], reverse=False)
  # print (id_label)
  with open(qgis_file_in2, 'w') as fout:
    fout.write('ID\tLabel\n')
    for idx, label in newid_label:
      temp_str = str(idx) + '\t' + str(label) + '\n'
      fout.write(temp_str)

  print("====done main_poaching_predict====")


def prep_qgis(qgis_file_in, qgis_file_out,
              cellsize, Xcorner, Ycorner,
              df_alldata):
  ''' translates output from main_poaching_predict into an ASC file.'''
  print("====start prep_qgis====")
  l_id = df_alldata["DN"].values
  l_X = df_alldata["X"].values
  l_Y = df_alldata["Y"].values

  if (len(l_id) != len(l_X)) or (len(l_X) != len(l_Y)):
    print("prep_qgis dim not match")

  ID_coordinate = dict()

  # (128.33402335, 43.6057568717)
  for i in range(0, len(l_id)):
    ID_coordinate[l_id[i]] = (l_X[i], l_Y[i])
    # print(ID_coordinate[l_id[i]])

  # Map configuration
  x_set = set()
  y_set = set()
  for index in ID_coordinate:
    x_set.add(ID_coordinate[index][0])
    y_set.add(ID_coordinate[index][1])
  min_x = int(min(x_set) / cellsize)
  min_y = int(min(y_set) / cellsize)
  max_x = int(max(x_set) / cellsize)
  max_y = int(max(y_set) / cellsize)

  print("min_x: ", min_x, " max_x: ", max_x,
        " min_y: ", min_y, " max_y: ", max_y)

  # TODO: I believe using only x_set is not enough here.
  # TODO: Should choose a max between x_set and y_set.
  dim = 1 + int((max(x_set) - min(x_set)) / cellsize)
  print(f'dim: {dim}')
  Map = np.zeros([dim, dim])

  # Load target list
  id_label = {}
  with open(qgis_file_in) as fin:
    fin.readline()
    for line in fin:
      line = line.strip().split()
      index = int(float(line[0]))
      label = float(line[1])
      id_label[index] = label
  '''id_label: {..., 3078: 0.006897479, 3079: 0.006897479, ...}'''

  valid = 0
  count = 0
  coincides = 0
  nearest_int = lambda x : int(round(x))
  for index in ID_coordinate:
    id_x = nearest_int((ID_coordinate[index][0] - min(x_set)) / cellsize)
    id_y = nearest_int((ID_coordinate[index][1] - min(y_set)) / cellsize)
    print(id_x, id_y, " -> ", id_label[index])
    try:
      valid += 1
      if Map[id_y, id_x] > 1E-20:
        coincides += 1
        Map[id_y, id_x + 1] = id_label[index]
      else:
        Map[id_y, id_x] = id_label[index]
    except:
      count += 1
      Map[id_y, id_x] = 0.0

  print("number of key error: %d" % count, "  number of valid: ",
        valid, "  number of coincides: ", coincides)

  with open(qgis_file_out, 'w') as fout:
    fout.write('NCOLS ' + str(dim) + '\n')
    fout.write('NROWS ' + str(dim) + '\n')
    fout.write('XLLCORNER ' + str(Xcorner) + '\n')
    fout.write('YLLCORNER ' + str(Ycorner) + '\n')
    fout.write('CELLSIZE ' + str(cellsize) + '\n')
    fout.write('NODATA_VALUE 0\n')
    info = ''
    for line in Map:
      info = ' '.join([str(x) for x in line]) + '\n' + info
    fout.write(info)

  print("====done prep_qgis====")


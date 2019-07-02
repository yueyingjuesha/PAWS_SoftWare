import numpy as np
import pandas as pd
from mydataset import DataSet
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.ensemble import BaggingClassifier
from sklearn import tree



def process_automate_data(files, column_names):

  raw_df_list = []
  for f in files:

    raw_df_list.append(pd.read_csv(f))


  DN_df = raw_df_list[0][['DN']].sort_values(by=['DN'])
  DN_df.reset_index(inplace=True)

  select_df_list = []
  for i in range(0, len(raw_df_list)):
    raw_df_list[i].columns = ['DN', column_names[i]]
    cur_sorted_df = raw_df_list[i].sort_values(by=['DN'])
    cur_sorted_df.reset_index(inplace=True)
    cur_select_df = cur_sorted_df[[column_names[i]]]

    cur_normalized_df = (cur_select_df - cur_select_df.min()) / \
        (cur_select_df.max() - cur_select_df.min())
    cur_normalized_df.columns = ["normal-" + column_names[i]]

    select_df_list.append(cur_select_df)
    if column_names[i][0:3] != 'is-':
      select_df_list.append(cur_normalized_df)


  select_df_list = [DN_df] + select_df_list

  comb_DN_ABC = pd.concat(select_df_list, axis=1)
  comb_DN_ABC.sort_values(by=["DN"], inplace=True)
  comb_DN_ABC.drop(['index'], axis=1)
  final_data = comb_DN_ABC
  return final_data


def preprocessing_fn1(fn1, patrol, poaching, selected_features):

  df_alldata = fn1
  df_validdata = df_alldata.dropna()
  df_invaliddata = df_alldata[df_alldata.isnull().any(axis=1)]
  df_knowndata = df_validdata[(df_validdata[patrol] > 0)]
  df_unknowndata = df_validdata[(df_validdata[patrol] == 0)]
  df_allpositive = df_knowndata[(df_knowndata[poaching] != 0)]
  df_allnegative = df_knowndata[(df_knowndata[poaching] == 0)]

  df_slct_positive = df_allpositive[selected_features]
  df_slct_negative = df_allnegative[selected_features]
  df_slct_unlabeled = df_unknowndata[selected_features]

  PositiveData = df_slct_positive.values
  NegativeData = df_slct_negative.values
  UnknownData = df_slct_unlabeled.values


  return df_alldata, df_invaliddata, df_unknowndata, df_allpositive, \
      df_allnegative, df_slct_positive, df_slct_negative, \
      df_slct_unlabeled, \
      PositiveData, NegativeData, UnknownData




def preprocessing_fn2(fn2, selected_features):

  df_alldata2 = fn2
  df_validdata2 = df_alldata2.dropna()
  df_invaliddata2 = df_alldata2[df_alldata2.isnull().any(axis=1)]
  df_slct_valid = df_validdata2[selected_features]
  NewAllData = df_slct_valid.values  
  return df_alldata2, df_validdata2, df_invaliddata2, df_slct_valid, NewAllData




def build_dataset(PositiveData, NegativeData, UnknownData, FoldNum=4):

  fold_pos_num = len(PositiveData) // FoldNum
  fold_neg_num = len(NegativeData) // FoldNum


  np.random.shuffle(NegativeData)
  neg = NegativeData[:fold_neg_num]
  NegativeData = NegativeData[fold_neg_num:]

  sample_size = NegativeData.shape[0]
  indx = np.random.randint(UnknownData.shape[0], size=sample_size)
  Udata = UnknownData[indx]


  NotFam = np.concatenate((Udata, NegativeData), axis=0)
  neg_label = np.array([0.] * len(neg))
  Fam = PositiveData

  dataset = DataSet(positive=Fam, negative=NotFam, fold_num=FoldNum)


  return neg, NegativeData, NotFam, neg_label, Fam, dataset



def main_poaching_predict(
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
    method='xgb',
):

  PositiveDataID = df_allpositive["DN"].values
  NegativeDataID = df_allnegative["DN"].values
  UnknownDataID = df_unknowndata["DN"].values

  ALLID = list(PositiveDataID) + list(NegativeDataID) + list(UnknownDataID)
  ALLDATA = list(df_slct_positive.values) + \
      list(df_slct_negative.values) + \
      list(df_slct_unlabeled.values)
  ALLDATA = np.array(ALLDATA)

  NEWALLID = list(df_validdata2["DN"].values)
  NEWALLDATA = list(df_slct_valid.values)
  NEWALLDATA = np.array(NEWALLDATA)

  if method == 'xgb':
    train_data, train_label = dataset.get_train_all_up(100)

    param = {
        'max_depth': 10,
        'eta': 0.1,
        'silent': 1,
        'objective': 'binary:logistic'
    }
    num_round = 1000
    D_train = xgb.DMatrix(train_data, label=train_label)

    bst = xgb.train(param, D_train, num_round)

    D_ALLDATA = xgb.DMatrix(ALLDATA)
    ALL_value = bst.predict(D_ALLDATA)

  elif method == 'dt':
    train_data, train_label = dataset.get_train_all_up(100)

    clf = BaggingClassifier(tree.DecisionTreeClassifier(
        criterion="entropy"), n_estimators=1000, max_samples=0.1)
    clf.fit(train_data, train_label)
    ALL_value = clf.predict_proba(ALLDATA)
    ALL_value = ALL_value[:, 1]

  elif method == 'svm':
    train_data, train_label = dataset.get_train_all_up(25)
    clf = SVR()
    clf.fit(train_data, train_label)
    ALL_value = clf.predict(ALLDATA)

  ALL_scores = np.zeros(len(ALL_value))  # Not used
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



  qgis_file_in1_str = ""
  for idx, label in id_label:
    temp_str = str(idx) + '\t' + str(label) + '\n'
    qgis_file_in1_str += temp_str



  if method == 'xgb':
    D_NEWALLDATA = xgb.DMatrix(NEWALLDATA)
    # prediction results
    ALL_newvalue = bst.predict(D_NEWALLDATA)

  elif method == 'dt':
    ALL_newvalue = clf.predict_proba(NEWALLDATA)
    ALL_newvalue = ALL_newvalue[:, 1]

  elif method == 'svm':
    ALL_newvalue = clf.predict(NEWALLDATA)

  ALL_newscores = np.zeros(len(ALL_newvalue))  # Not used
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

  return qgis_file_in1_str


def prep_qgis(qgis_file_in1_str, qgis_file_out,
              cellsize, Xcorner, Ycorner,
              df_alldata):

  l_id = df_alldata["DN"].values
  l_X = df_alldata["X"].values
  l_Y = df_alldata["Y"].values



  ID_coordinate = dict()

  for i in range(0, len(l_id)):
    ID_coordinate[l_id[i]] = (l_X[i], l_Y[i])

  x_set = set()
  y_set = set()
  for index in ID_coordinate:
    x_set.add(ID_coordinate[index][0])
    y_set.add(ID_coordinate[index][1])
  min_x = int(min(x_set) / cellsize)
  min_y = int(min(y_set) / cellsize)
  max_x = int(max(x_set) / cellsize)
  max_y = int(max(y_set) / cellsize)


  dimx = 1 + (max_x - min_x)
  dimy = 1 + (max_y - min_y)

  Map = np.zeros([dimy, dimx])

  # Load target list
  id_label = {}

  for line in qgis_file_in1_str.split('\n'):
    if len(line) == 0:
      continue
    line = line.strip().split()
    index = int(line[0])
    label = float(line[1])
    id_label[index] = label


  valid = 0
  count = 0
  coincides = 0
  nearest_int = lambda x: int(round(x))
  for index in ID_coordinate:
    id_x = nearest_int((ID_coordinate[index][0] - min(x_set)) / cellsize)
    id_y = nearest_int((ID_coordinate[index][1] - min(y_set)) / cellsize)

    valid += 1
    if Map[id_y, id_x] > 1E-20:
      coincides += 1
    else:
      Map[id_y, id_x] = id_label[index]

  with open(qgis_file_out, 'w') as fout:
    fout.write('NCOLS ' + str(dimx) + '\n')
    fout.write('NROWS ' + str(dimy) + '\n')
    fout.write('XLLCORNER ' + str(Xcorner) + '\n')
    fout.write('YLLCORNER ' + str(Ycorner) + '\n')
    fout.write('CELLSIZE ' + str(cellsize) + '\n')
    fout.write('NODATA_VALUE 0\n')
    info = ''
    for line in Map:
      info = ' '.join([str(x) for x in line]) + '\n' + info
    fout.write(info)
  return
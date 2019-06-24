import sys, os
from os import getcwd, system
from os.path import join


import numpy as np
import pandas as pd
import sys

from sklearn.metrics import roc_auc_score

import xgboost as xgb
import numpy as np
import pdb
import time

from pandas import read_csv, DataFrame


from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QComboBox, QVBoxLayout, QFileDialog, QMessageBox


class DataSet(object):
  '''How to sample data is important.'''
  def __init__(self, positive, negative, fold_num):
    '''Prepare data. Several folds for pos and neg.'''
    self.positive = np.array(positive)
    self.negative = np.array(negative)
    self.fold_num = fold_num

    index = np.random.permutation(len(self.positive))
    self.positive = self.positive[index]
    index = np.random.permutation(len(self.negative))
    self.negative = self.negative[index]

    # Split k-fold
    self.data_folds = []
    self.label_folds = []
    fold_pos_num = int(len(self.positive) / int(fold_num))
    fold_neg_num = int(len(self.negative) / int(fold_num))

    for i in range(fold_num):
      if i == fold_num - 1:
        pos = self.positive[i * fold_pos_num:]
        neg = self.negative[i * fold_neg_num:]

        data = np.concatenate((pos, neg), axis=0)
        label = np.array([1.] * len(pos) + [0.] * len(neg))

        index = np.random.permutation(len(data))
        self.data_folds.append(data[index])
        self.label_folds.append(label[index])
      else:
        pos = self.positive[i * fold_pos_num:(i + 1) * fold_pos_num]
        neg = self.negative[i * fold_neg_num:(i + 1) * fold_neg_num]

        data = np.concatenate((pos, neg), axis=0)
        label = np.array([1.] * len(pos) + [0.] * len(neg))

        index = np.random.permutation(len(data))
        self.data_folds.append(data[index])
        self.label_folds.append(label[index])

  def update_negative(self, negative):
    self.negative = np.array(negative)
    index = np.random.permutation(len(self.negative))
    self.negative = self.negative[index]
    fold_num = self.fold_num
    self.data_folds = []
    self.label_folds = []
    fold_pos_num = int(len(self.positive) / int(fold_num))
    fold_neg_num = int(len(self.negative) / int(fold_num))
    for i in range(fold_num):
      if i == fold_num - 1:
        pos = self.positive[i * fold_pos_num:]
        neg = self.negative[i * fold_neg_num:]

        data = np.concatenate((pos, neg), axis=0)
        label = np.array([1.] * len(pos) + [0.] * len(neg))

        index = np.random.permutation(len(data))
        self.data_folds.append(data[index])
        self.label_folds.append(label[index])
      else:
        pos = self.positive[i * fold_pos_num:(i + 1) * fold_pos_num]
        neg = self.negative[i * fold_neg_num:(i + 1) * fold_neg_num]

        data = np.concatenate((pos, neg), axis=0)
        label = np.array([1.] * len(pos) + [0.] * len(neg))

        index = np.random.permutation(len(data))
        self.data_folds.append(data[index])
        self.label_folds.append(label[index])

  def get_train_test(self, fold_id):
    data_folds_copy = list(self.data_folds)
    label_folds_copy = list(self.label_folds)

    test_data = data_folds_copy.pop(fold_id)
    test_label = label_folds_copy.pop(fold_id)

    train_data = np.concatenate(data_folds_copy, axis=0)
    train_label = np.concatenate(label_folds_copy, axis=0)
    return train_data, train_label, test_data, test_label

  def get_train_test_upsample(self, fold_id, num):
    data_folds_copy = list(self.data_folds)
    label_folds_copy = list(self.label_folds)

    test_data = data_folds_copy.pop(fold_id)
    test_label = label_folds_copy.pop(fold_id)

    train_data = np.concatenate(data_folds_copy, axis=0)
    train_label = np.concatenate(label_folds_copy, axis=0)

    train_data_up = []
    train_label_up = []
    for data, label in zip(train_data, train_label):
      if label:
        train_data_up += [data] * num
        train_label_up += [label] * num
      else:
        train_data_up.append(data)
        train_label_up.append(label)

    train_data_up = np.array(train_data_up)
    train_label_up = np.array(train_label_up)
    index = np.random.permutation(len(train_label_up))
    train_data_up = train_data_up[index]
    train_label_up = train_label_up[index]

    return train_data_up, train_label_up, test_data, test_label

  def get_train_all(self):
    data_folds_copy = list(self.data_folds)
    label_folds_copy = list(self.label_folds)

    train_data = np.concatenate(data_folds_copy, axis=0)
    train_label = np.concatenate(label_folds_copy, axis=0)
    return train_data, train_label

  def get_train_all_up(self, num):
    data_folds_copy = list(self.data_folds)
    label_folds_copy = list(self.label_folds)

    train_data = np.concatenate(data_folds_copy, axis=0)
    train_label = np.concatenate(label_folds_copy, axis=0)

    train_data_up = []
    train_label_up = []
    for data, label in zip(train_data, train_label):
      if label:
        train_data_up += [data] * num
        train_label_up += [label] * num
      else:
        train_data_up.append(data)
        train_label_up.append(label)

    train_data_up = np.array(train_data_up)
    train_label_up = np.array(train_label_up)
    index = np.random.permutation(len(train_label_up))
    train_data_up = train_data_up[index]
    train_label_up = train_label_up[index]

    return train_data_up, train_label_up

  def get_train_neg_traintest_pos(self, fold_id, num):
    #start = time.time()
    data_folds_copy = list(self.data_folds)
    label_folds_copy = list(self.label_folds)

    test_data1 = data_folds_copy.pop(fold_id)
    test_label1 = label_folds_copy.pop(fold_id)

    train_data = np.concatenate(data_folds_copy, axis=0)
    train_label = np.concatenate(label_folds_copy, axis=0)

    train_data_up = []
    train_label_up = []
    test_data = []
    test_label = []
    for data, label in zip(train_data, train_label):
      if label:
        train_data_up += [data] * num
        train_label_up += [label] * num
      else:
        train_data_up.append(data)
        train_label_up.append(label)
    for data, label in zip(test_data1, test_label1):
      if label:
        test_data.append(data)
        test_label.append(label)
      else:
        train_data_up.append(data)
        train_label_up.append(label)

    train_data_up = np.array(train_data_up)
    train_label_up = np.array(train_label_up)
    index = np.random.permutation(len(train_label_up))
    train_data_up = train_data_up[index]
    train_label_up = train_label_up[index]
    #print("duration" , time.time() - start)
    return train_data_up, train_label_up, test_data, test_label

  def get_train_neg_traintest_pos_smote(self, fold_id, num):
    #start = time.time()
    data_folds_copy = list(self.data_folds)
    label_folds_copy = list(self.label_folds)

    test_data1 = data_folds_copy.pop(fold_id)
    test_label1 = label_folds_copy.pop(fold_id)

    train_data = np.concatenate(data_folds_copy, axis=0)
    train_label = np.concatenate(label_folds_copy, axis=0)

    train_data_up = []
    train_label_up = []
    test_data = []
    test_label = []
    train_data_pos = []
    train_label_pos = []
    for data, label in zip(train_data, train_label):
      if label:
        train_data_pos += [data]
        train_label_pos += [label]
      else:
        train_data_up.append(data)
        train_label_up.append(label)
    for data, label in zip(test_data1, test_label1):
      if label:
        test_data.append(data)
        test_label.append(label)
      else:
        train_data_up.append(data)
        train_label_up.append(label)

    train_data_pos = np.array(train_data_pos)
    train_label_pos = np.array(train_label_pos)
    # pdb.set_trace()
    idx_sort = np.argsort(np.sum(np.square(
        np.expand_dims(train_data_pos, 2) - np.tile(train_data_pos, (train_data_pos.shape[0], 1)).reshape(train_data_pos.shape + (train_data_pos.shape[0],))), axis=1), axis=1)
    for j, (data, label) in enumerate(zip(train_data_pos, train_label_pos)):
      for i in range(num):
        a = np.random.uniform(0, 1)
        idx = np.random.randint(len(train_data_pos))
        train_data_up += [data * a +
                          (1 - a) * train_data_pos[idx_sort[j, idx]]]
        train_label_up += [label]

    train_data_up = np.array(train_data_up)
    train_label_up = np.array(train_label_up)
    index = np.random.permutation(len(train_label_up))
    train_data_up = train_data_up[index]
    train_label_up = train_label_up[index]
    #print("smote_duration" , time.time() - start)
    return train_data_up, train_label_up, test_data, test_label

  def get_train_all_up_aug(self, UdataPos, num):
    data_folds_copy = list(self.data_folds)
    label_folds_copy = list(self.label_folds)

    train_data = np.concatenate(data_folds_copy, axis=0)
    train_label = np.concatenate(label_folds_copy, axis=0)
    # UdataPos = []
    # UdataPos1 = []
    '''
        for i,id in enumerate(UnknownDataID.reshape(-1)):
            if (id in cluster_ids[8] or id in cluster_ids[7]) and (id in cluster_ids50[7] or id in cluster_ids50[6]):
                UdataPos.append(UnknownData[i:i+1,:])
        #        UdataPos.append(UnknownData[i:i+1,:])
        '''
    # UdataPos.append(UnknownData[index[:1000]])
    # for i,id in enumerate(UnknownDataID.reshape(-1)):
    #     if (id in cluster_ids[8] or id in cluster_ids[7] or id in cluster_ids[6]) and (id in cluster_ids50[7] or id in cluster_ids50[6]):
    #         UdataPos.append(UnknownData[i:i+1,:])
    # #        UdataPos.append(UnknownData[i:i+1,:])
    # UdataPos=np.concatenate(UdataPos,0)
    # print(len(UdataPos))

    train_data_up = []
    train_label_up = []
    for data, label in zip(train_data, train_label):
      if label:
        train_data_up += [data] * num
        train_label_up += [label] * num
        label1 = label
      else:
        train_data_up.append(data)
        train_label_up.append(label)
    for data in UdataPos:
      train_data_up += [data]
      train_label_up += [label1]

    train_data_up = np.array(train_data_up)
    train_label_up = np.array(train_label_up)
    index = np.random.permutation(len(train_label_up))
    train_data_up = train_data_up[index]
    train_label_up = train_label_up[index]

    return train_data_up, train_label_up

  def get_train_neg_traintest_pos_aug(self, cluster_ids, cluster_ids50, index, UnknownData, UnknownDataID, fold_id, num):
    data_folds_copy = list(self.data_folds)
    label_folds_copy = list(self.label_folds)

    test_data1 = data_folds_copy.pop(fold_id)
    test_label1 = label_folds_copy.pop(fold_id)

    train_data = np.concatenate(data_folds_copy, axis=0)
    train_label = np.concatenate(label_folds_copy, axis=0)

    UdataPos = []
    '''
        for i,id in enumerate(UnknownDataID.reshape(-1)):
            if (id in cluster_ids[8] or id in cluster_ids[7]) and (id in cluster_ids50[7] or id in cluster_ids50[6]):
                UdataPos.append(UnknownData[i:i+1,:])
        #        UdataPos.append(UnknownData[i:i+1,:])
        '''
    # UdataPos.append(UnknownData[index[:1000]])
    for i, id in enumerate(UnknownDataID.reshape(-1)):
      if (id in cluster_ids[8] or id in cluster_ids[7] or id in cluster_ids[6]) and (id in cluster_ids50[7] or id in cluster_ids50[6]):
        UdataPos.append(UnknownData[i:i + 1, :])
    #        UdataPos.append(UnknownData[i:i+1,:])
    UdataPos = np.concatenate(UdataPos, 0)
    print(UdataPos.shape)
    train_data_up = []
    train_label_up = []
    test_data = []
    test_label = []
    for data, label in zip(train_data, train_label):
      if label:
        train_data_up += [data] * num
        train_label_up += [label] * num
        label1 = label
      else:
        train_data_up.append(data)
        train_label_up.append(label)
    for data, label in zip(test_data1, test_label1):
      if label:
        test_data.append(data)
        test_label.append(label)
      else:
        train_data_up.append(data)
        train_label_up.append(label)
    for data in UdataPos:
      train_data_up += [data]
      train_label_up += [label1]
    train_data_up = np.array(train_data_up)
    train_label_up = np.array(train_label_up)
    index = np.random.permutation(len(train_label_up))
    train_data_up = train_data_up[index]
    train_label_up = train_label_up[index]
    return train_data_up, train_label_up, test_data, test_label


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





def load_csv(path):
    file = read_csv(path)
    return file.values

def save_csv(data, path):
    DataFrame(data).to_csv(path)

def count(data):
    return data ** 2

def calculate_sum(data):
    return data ** 2


class toy_runner():
    def __init__(self, mode, path):
        self.mode = mode
        #self.data = load_csv(path)
        self.path = path
        self.result = None
        self.run_flag = False

        self.warm_message = None

    def run(self):
        if self.mode == 1:
            result = run_makedata(self.path)
            if result == 'Finished!':
                self.run_flag = True
            else:
                self.run_flag = False
                self.warm_message = file
        elif self.mode == 2:
            self.result = calculate_sum(load_csv(path))
            self.run_flag = True

    def check_result(self):
        return self.run_flag

    def get_result(self):
        return self.result

    def save_result(self, dir):
        path = join(dir, "toy_output.csv")
        save_csv(self.result, path)


class MainForm(QWidget):
    def __init__(self, name = 'MainForm'):
        super(MainForm,self).__init__()
        self.setWindowTitle(name)
        self.cwd = getcwd() 
        self.resize(300, 100) 

        self.chosen_model = None
        self.has_result = None
        self.chosen_file = None
        self.save_path = None
        self.runner = None



        ## btn
        self.label1 = QLabel("Choose File:", self)
        self.btn_chooseFile = QPushButton("Choose File", self)  

        self.label2 = QLabel("Select Model:", self)
        self.btn_selectModel = QComboBox(self)  
        self.btn_selectModel.addItems(['model1','model2','model3'])

        self.btn_runModel = QPushButton("Run Model", self) 

        self.label3 = QLabel("Choose Save Path:", self)
        self.btn_chooseDir = QPushButton("Choose Save Path", self)  

        self.btn_exportResult = QPushButton("Export Result", self)  

        self.btn_runModel.setEnabled(False)
        self.btn_exportResult.setEnabled(False)





        layout = QVBoxLayout()
        layout.addWidget(self.label1)
        layout.addWidget(self.btn_chooseFile)
        layout.addWidget(self.label2)
        layout.addWidget(self.btn_selectModel)
        layout.addWidget(self.btn_runModel)
        layout.addWidget(self.label3)
        layout.addWidget(self.btn_chooseDir)
        layout.addWidget(self.btn_exportResult)
        self.setLayout(layout)


        self.btn_chooseFile.clicked.connect(self.slot_btn_chooseFile)
        self.btn_selectModel.activated[str].connect(self.slot_btn_selectModel)
        self.btn_runModel.clicked.connect(self.slot_btn_runModel)
        self.btn_chooseDir.clicked.connect(self.slot_btn_chooseDir)
        self.btn_exportResult.clicked.connect(self.slot_btn_exportResult)





    def slot_btn_chooseFile(self):
        self.chosen_file = QFileDialog.getExistingDirectory(self, "getExistingDirectory", "./") 
        self.btn_chooseFile.setText(self.chosen_file)
        if self.chosen_model:
            self.btn_runModel.setEnabled(True)
        return


    def slot_btn_selectModel(self, text):
        self.chosen_model = text
        if self.chosen_file:
            self.btn_runModel.setEnabled(True)
        return


    def slot_btn_runModel(self):
        QMessageBox.information(self, 'info1', 'Running {}, please wait'.format(self.chosen_model))
        model_type = int(self.chosen_model[-1])
        self.runner = toy_runner(model_type, self.chosen_file)
        self.runner.run()
        if self.runner.warm_message:
            QMessageBox.information(self, 'info3', 'No such a file in selected path: {}'.format(self.runner.warm_message))
            return 
        QMessageBox.information(self, 'info0', 'Running finished!'.format(self.chosen_model))
        self.has_result = self.runner.check_result()
        if self.save_path and self.has_result:
            self.btn_exportResult.setEnabled(True)
        return

    def slot_btn_exportResult(self):
        QMessageBox.information(self, 'info2', 'Results are saved in \n{}'.format(self.save_path))
        if self.chosen_model == 'model1':
            system('move final.csv {}'.format(self.save_path))
            system('move predictions1.txt {}'.format(self.save_path))
            system('move predictions2.txt {}'.format(self.save_path))
            system('move predictions_heatmap1.asc {}'.format(self.save_path))
        else:
            self.runner.save_result(self.save_path)
        return

    def slot_btn_chooseDir(self):
        dir_choose = QFileDialog.getExistingDirectory(self,  
                                    "Choose Path",  
                                    self.cwd) 

        self.save_path = dir_choose
        self.btn_chooseDir.setText(dir_choose)
        if self.save_path and self.has_result:
            self.btn_exportResult.setEnabled(True)
        return




    def closeEvent(self, event):  
        reply = QMessageBox.question(self,
                                               'exit',
                                               "Do you want to exit?",
                                               QMessageBox.Yes | QMessageBox.No,
                                               QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

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

  process_automate_data(files, column_names)

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
      preprocessing_fn1(
          fn1, patrol, poaching, selected_features)

  df_alldata2, df_validdata2, df_invaliddata2, df_slct_valid, NewAllData = \
      preprocessing_fn2(fn2, selected_features)

  FoldNum = 4
  neg, NegativeData, NotFam, neg_label, Fam, dataset = \
      build_dataset(
          PositiveData, NegativeData, UnknownData, FoldNum=FoldNum)

  ##########################################################################

  main_poaching_predict(qgis_file_in1, qgis_file_in2,
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
  prep_qgis(qgis_file_in1, qgis_file_out1, gridDim1,
                             xcorner1, ycorner1, df_alldata)
  return 'Finished!'




if __name__=="__main__":
    app = QApplication(sys.argv)
    mainForm = MainForm('Demo V1.0')
    mainForm.show()
    sys.exit(app.exec_())

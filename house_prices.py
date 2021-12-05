"""
Created on Fri Dec  3 21:57:58 2021

@author: Christian
"""
import numpy as np
np.set_printoptions(suppress = True, linewidth = 100, precision = 2)

# revision de datos 
raw_data = np.genfromtxt("C:\\Users\\Christian\\Documents\\Python_courses\\house_prices\\train.csv", 
                         delimiter = ',', 
                         dtype = np.str_, 
                         autostrip = True) 
raw_data.shape

# eliminar titulos
raw_data = np.genfromtxt("C:\\Users\\Christian\\Documents\\Python_courses\\house_prices\\train.csv", 
                         delimiter = ',', 
                         autostrip = True,
                         skip_header = 1)
raw_data

# revisar datos nullos (str)
np.isnan(raw_data).sum()

# estadisticas de relleno
temporary_fill = np.nanmax(raw_data) + 1
temporary_mean = np.nanmean(raw_data, axis = 0)

# son 43 columnas str
np.isnan(temporary_mean).sum()

# Resto de estadisticas
temporary_stats = np.array([np.nanmin(raw_data, axis = 0),
                           temporary_mean,
                           np.nanmax(raw_data, axis = 0)])
temporary_stats

# variables numericas y no numericas
str_cols = np.argwhere(np.isnan(temporary_mean)).squeeze()
str_cols.shape

nums_cols = np.argwhere(np.isnan(temporary_mean) == False).squeeze()
nums_cols.shape

str_data_train = np.genfromtxt("C:\\Users\\Christian\\Documents\\Python_courses\\house_prices\\train.csv", 
                         delimiter = ',', 
                         autostrip = True,
                         skip_header = 1,
                         usecols = str_cols,
                         dtype = np.str_)
str_data_train

num_data_train = np.genfromtxt("C:\\Users\\Christian\\Documents\\Python_courses\\house_prices\\train.csv", 
                         delimiter = ',', 
                         autostrip = True,
                         skip_header = 1,
                         usecols = nums_cols)
num_data_train.shape

#========================================================================================================================
# Lo mismo pero para el test
#========================================================================================================================
raw_data_test = np.genfromtxt("C:\\Users\\Christian\\Documents\\Python_courses\\house_prices\\test.csv", 
                         delimiter = ',', 
                         autostrip = True,
                         skip_header = 1)

temporary_mean_test = np.nanmean(raw_data_test, axis = 0)
srt_cols_test = np.argwhere(np.isnan(temporary_mean_test)).squeeze()
nums_cols_test = np.argwhere(np.isnan(temporary_mean_test) == False).squeeze()

num_data_test = np.genfromtxt("C:\\Users\\Christian\\Documents\\Python_courses\\house_prices\\test.csv", 
                         delimiter = ',', 
                         autostrip = True,
                         skip_header = 1,
                         usecols = nums_cols_test)

str_data_test = np.genfromtxt("C:\\Users\\Christian\\Documents\\Python_courses\\house_prices\\test.csv", 
                         delimiter = ',', 
                         autostrip = True,
                         skip_header = 1,
                         usecols = srt_cols_test, 
                         dtype = np.str_)

names_full_train = raw_data = np.genfromtxt("C:\\Users\\Christian\\Documents\\Python_courses\\house_prices\\train.csv", 
                         delimiter = ',',
                         skip_footer = str_data_train.shape[0],
                         dtype = np.str_, 
                         autostrip = True) 
names_full_train

names_full_test = raw_data = np.genfromtxt("C:\\Users\\Christian\\Documents\\Python_courses\\house_prices\\train.csv", 
                         delimiter = ',',
                         skip_footer = str_data_train.shape[0],
                         dtype = np.str_, 
                         autostrip = True) 
names_full_test = np.genfromtxt("C:\\Users\\Christian\\Documents\\Python_courses\\house_prices\\test.csv", 
                         delimiter = ',', 
                         autostrip = True,
                         skip_footer = str_data_test.shape[0], 
                         dtype = np.str_)
names_full_test

names_num_train, names_str_train = names_full_train[nums_cols], names_full_train[str_cols]

names_num_test, names_str_test = names_full_test[nums_cols_test], names_full_test[srt_cols_test]


# primeras cols str
names_str_train[0]
names_str_test[0]
str_data_train[:, 0]
str_data_test[:, 0]
np.array(np.unique(str_data_train[:, 0], return_counts = True))
np.array(np.unique(str_data_test[:, 0], return_counts = True))
dat_str = np.array(['C (all)', 'FV', 'NA', 'RH', 'RL', 'RM'])

for i in range(0, len(dat_str)):
    str_data_train[:, 0] = np.where(str_data_train[:, 0] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 0])

for i in range(0, len(dat_str)):
    str_data_test[:, 0] = np.where(str_data_test[:, 0] == dat_str[i], 
                                       i + 1, 
                                       str_data_test[:, 0])

# segundas cols str
names_str_train[1]
names_str_test[1]
str_data_train[:, 1]
str_data_test[:, 1]
np.array(np.unique(str_data_train[:, 1], return_counts = True))
np.array(np.unique(str_data_test[:, 1], return_counts = True))

str_data_train[:, 1] = np.where(str_data_train[:, 1] == 'Grvl', 
                                       1, 0)

str_data_test[:, 1] = np.where(str_data_test[:, 1] == 'Grvl', 
                                       1, 0)

# otra
names_str_train[2]
names_str_test[2]
str_data_train[:, 2]
str_data_test[:, 2]
np.array(np.unique(str_data_train[:, 2], return_counts = True))
np.array(np.unique(str_data_test[:, 2], return_counts = True))
dat_str = np.array(['Grvl', 'NA', 'Pave'])

for i in range(0, len(dat_str)):
    str_data_train[:, 2] = np.where(str_data_train[:, 2] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 2])

for i in range(0, len(dat_str)):
    str_data_test[:, 2] = np.where(str_data_test[:, 2] == dat_str[i], 
                                       i + 1, 
                                       str_data_test[:, 2])
    
# 4
np.array(np.unique(str_data_train[:, 3], return_counts = True))
np.array(np.unique(str_data_test[:, 3], return_counts = True))
dat_str = np.array(['IR1', 'IR2', 'IR3', 'Reg'])

for i in range(0, len(dat_str)):
    str_data_train[:, 3] = np.where(str_data_train[:, 3] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 3])

for i in range(0, len(dat_str)):
    str_data_test[:, 3] = np.where(str_data_test[:, 3] == dat_str[i], 
                                       i + 1, 
                                       str_data_test[:, 3])

# 5
names_str_train[4]
names_str_test[4]
str_data_train[:, 4]
str_data_test[:, 4]
np.array(np.unique(str_data_train[:, 4], return_counts = True))
np.array(np.unique(str_data_test[:, 4], return_counts = True))
dat_str = np.array(['Bnk', 'HLS', 'Low', 'Lvl'])

for i in range(0, len(dat_str)):
    str_data_train[:, 4] = np.where(str_data_train[:, 4] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 4])

for i in range(0, len(dat_str)):
    str_data_test[:, 4] = np.where(str_data_test[:, 4] == dat_str[i], 
                                       i + 1, 
                                       str_data_test[:, 4])

# 6
np.array(np.unique(str_data_train[:, 5], return_counts = True))
np.array(np.unique(str_data_test[:, 5], return_counts = True))

str_data_train[:, 5] = np.where(str_data_train[:, 5] == 'AllPub', 
                                       1, 0)

str_data_test[:, 5] = np.where(str_data_test[:, 5] == 'AllPub', 
                                       1, 0)

# 7
np.array(np.unique(str_data_train[:, 6], return_counts = True))
np.array(np.unique(str_data_test[:, 6], return_counts = True))
dat_str = np.array(['Corner', 'CulDSac', 'FR2', 'FR3', 'Inside'])

for i in range(0, len(dat_str)):
    str_data_train[:, 6] = np.where(str_data_train[:, 6] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 6])

for i in range(0, len(dat_str)):
    str_data_test[:, 6] = np.where(str_data_test[:, 6] == dat_str[i], 
                                       i + 1, 
                                       str_data_test[:, 6])
    
# 8 
np.array(np.unique(str_data_train[:, 7], return_counts = True))
np.array(np.unique(str_data_test[:, 7], return_counts = True))
dat_str = np.array(['Gtl', 'Mod', 'Sev'])

for i in range(0, len(dat_str)):
    str_data_train[:, 7] = np.where(str_data_train[:, 7] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 7])

for i in range(0, len(dat_str)):
    str_data_test[:, 7] = np.where(str_data_test[:, 7] == dat_str[i], 
                                       i + 1, 
                                       str_data_test[:, 7])
    
# 9
np.array(np.unique(str_data_train[:, 8], return_counts = True))
np.array(np.unique(str_data_test[:, 8], return_counts = True))
dat_str = np.array(['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards',
        'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NPkVill', 'NWAmes', 'NoRidge',
        'NridgHt', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber',
        'Veenker'])

for i in range(0, len(dat_str)):
    str_data_train[:, 8] = np.where(str_data_train[:, 8] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 8])

for i in range(0, len(dat_str)):
    str_data_test[:, 8] = np.where(str_data_test[:, 8] == dat_str[i], 
                                       i + 1, 
                                       str_data_test[:, 8])

# 10
np.array(np.unique(str_data_train[:, 9], return_counts = True))
np.array(np.unique(str_data_test[:, 9], return_counts = True))
dat_str = np.array(['Artery', 'Feedr', 'Norm', 'PosA', 'PosN', 'RRAe', 'RRAn', 'RRNe', 'RRNn'])

for i in range(0, len(dat_str)):
    str_data_train[:, 9] = np.where(str_data_train[:, 9] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 9])

for i in range(0, len(dat_str)):
    str_data_test[:, 9] = np.where(str_data_test[:, 9] == dat_str[i], 
                                       i + 1, 
                                       str_data_test[:, 9])

# 11 
names_str_train[10]
names_str_test[10]
np.array(np.unique(str_data_train[:, 10], return_counts = True))
np.array(np.unique(str_data_test[:, 10], return_counts = True))
dat_str = np.array(['Artery', 'Feedr', 'Norm', 'PosA', 'PosN', 'RRAe', 'RRAn', 'RRNn'])

for i in range(0, len(dat_str)):
    str_data_train[:, 10] = np.where(str_data_train[:, 10] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 10])

for i in range(0, len(dat_str)):
    str_data_test[:, 10] = np.where(str_data_test[:, 10] == dat_str[i], 
                                       i + 1, 
                                       str_data_test[:, 10])

# 12
np.array(np.unique(str_data_train[:, 11], return_counts = True))
np.array(np.unique(str_data_test[:, 11], return_counts = True))
dat_str = np.array(['1Fam', '2fmCon', 'Duplex', 'Twnhs', 'TwnhsE'])

for i in range(0, len(dat_str)):
    str_data_train[:, 11] = np.where(str_data_train[:, 11] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 11])

for i in range(0, len(dat_str)):
    str_data_test[:, 11] = np.where(str_data_test[:, 11] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 11])

# 13 
np.array(np.unique(str_data_train[:, 12], return_counts = True))
np.array(np.unique(str_data_test[:, 12], return_counts = True))
dat_str = np.array(['1.5Fin', '1.5Unf', '1Story', '2.5Fin', '2.5Unf', '2Story', 'SFoyer', 'SLvl'])

for i in range(0, len(dat_str)):
    str_data_train[:, 12] = np.where(str_data_train[:, 12] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 12])

for i in range(0, len(dat_str)):
    str_data_test[:, 12] = np.where(str_data_test[:, 12] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 12])

# 14
np.array(np.unique(str_data_train[:, 13], return_counts = True))
np.array(np.unique(str_data_test[:, 13], return_counts = True))
dat_str = np.array(['Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'])

for i in range(0, len(dat_str)):
    str_data_train[:, 13] = np.where(str_data_train[:, 13] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 13])

for i in range(0, len(dat_str)):
    str_data_test[:, 13] = np.where(str_data_test[:, 13] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 13])

# 15
np.array(np.unique(str_data_train[:, 14], return_counts = True))
np.array(np.unique(str_data_test[:, 14], return_counts = True))
dat_str = np.array(['ClyTile', 'CompShg', 'Membran', 'Metal', 'Roll', 'Tar&Grv', 'WdShake', 'WdShngl'])

for i in range(0, len(dat_str)):
    str_data_train[:, 14] = np.where(str_data_train[:, 14] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 14])

for i in range(0, len(dat_str)):
    str_data_test[:, 14] = np.where(str_data_test[:, 14] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 14])

# 16
np.array(np.unique(str_data_train[:, 15], return_counts = True))
np.array(np.unique(str_data_test[:, 15], return_counts = True))
dat_str = np.array(['AsbShng', 'AsphShn', 'BrkComm', 'BrkFace', 'CBlock', 'CemntBd', 'HdBoard', 'ImStucc',
        'MetalSd', 'Plywood', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'WdShing', 'NA'])

for i in range(0, len(dat_str)):
    str_data_train[:, 15] = np.where(str_data_train[:, 15] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 15])

for i in range(0, len(dat_str)):
    str_data_test[:, 15] = np.where(str_data_test[:, 15] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 15])

# 17
np.array(np.unique(str_data_train[:, 16], return_counts = True))
np.array(np.unique(str_data_test[:, 16], return_counts = True))
dat_str = np.array(['AsbShng', 'AsphShn', 'Brk Cmn', 'BrkFace', 'CBlock', 'CmentBd', 'HdBoard', 'ImStucc',
        'MetalSd', 'Other', 'Plywood', 'Stone', 'Stucco', 'VinylSd', 'Wd Sdng', 'Wd Shng', 'NA'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 16] = np.where(str_data_train[:, 16] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 16])

for i in range(0, len(dat_str)):
    str_data_test[:, 16] = np.where(str_data_test[:, 16] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 16])

# 18
np.array(np.unique(str_data_train[:, 17], return_counts = True))
np.array(np.unique(str_data_test[:, 17], return_counts = True))
dat_str = np.array(['BrkCmn', 'BrkFace', 'NA', 'None', 'Stone'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 17] = np.where(str_data_train[:, 17] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 17])

for i in range(0, len(dat_str)):
    str_data_test[:, 17] = np.where(str_data_test[:, 17] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 17])

# 19
np.array(np.unique(str_data_train[:, 18], return_counts = True))
np.array(np.unique(str_data_test[:, 18], return_counts = True))
dat_str = np.array(['Ex', 'Fa', 'Gd', 'TA'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 18] = np.where(str_data_train[:, 18] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 18])

for i in range(0, len(dat_str)):
    str_data_test[:, 18] = np.where(str_data_test[:, 18] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 18])

# 20
np.array(np.unique(str_data_train[:, 19], return_counts = True))
np.array(np.unique(str_data_test[:, 19], return_counts = True))
dat_str = np.array(['Ex', 'Fa', 'Gd', 'Po', 'TA'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 19] = np.where(str_data_train[:, 19] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 19])

for i in range(0, len(dat_str)):
    str_data_test[:, 19] = np.where(str_data_test[:, 19] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 19])

# 21
np.array(np.unique(str_data_train[:, 20], return_counts = True))
np.array(np.unique(str_data_test[:, 20], return_counts = True))
dat_str = np.array(['BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 20] = np.where(str_data_train[:, 20] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 20])

for i in range(0, len(dat_str)):
    str_data_test[:, 20] = np.where(str_data_test[:, 20] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 20])

# 22
np.array(np.unique(str_data_train[:, 21], return_counts = True))
np.array(np.unique(str_data_test[:, 21], return_counts = True))
dat_str = np.array(['Ex', 'Fa', 'Gd', 'NA', 'TA'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 21] = np.where(str_data_train[:, 21] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 21])

for i in range(0, len(dat_str)):
    str_data_test[:, 21] = np.where(str_data_test[:, 21] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 21])

# 23 
np.array(np.unique(str_data_train[:, 22], return_counts = True))
np.array(np.unique(str_data_test[:, 22], return_counts = True))
dat_str = np.array(['Fa', 'Gd', 'NA', 'Po', 'TA'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 22] = np.where(str_data_train[:, 22] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 22])

for i in range(0, len(dat_str)):
    str_data_test[:, 22] = np.where(str_data_test[:, 22] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 22])

# 24
np.array(np.unique(str_data_train[:, 23], return_counts = True))
np.array(np.unique(str_data_test[:, 23], return_counts = True))
dat_str = np.array(['Av', 'Gd', 'Mn', 'NA', 'No'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 23] = np.where(str_data_train[:, 23] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 23])

for i in range(0, len(dat_str)):
    str_data_test[:, 23] = np.where(str_data_test[:, 23] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 23])

# 25
np.array(np.unique(str_data_train[:, 24], return_counts = True))
np.array(np.unique(str_data_test[:, 24], return_counts = True))
dat_str = np.array(['ALQ', 'BLQ', 'GLQ', 'LwQ', 'NA', 'Rec', 'Unf'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 24] = np.where(str_data_train[:, 24] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 24])

for i in range(0, len(dat_str)):
    str_data_test[:, 24] = np.where(str_data_test[:, 24] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 24])

# 26
np.array(np.unique(str_data_train[:, 25], return_counts = True))
np.array(np.unique(str_data_test[:, 25], return_counts = True))
dat_str = np.array(['ALQ', 'BLQ', 'GLQ', 'LwQ', 'NA', 'Rec', 'Unf'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 25] = np.where(str_data_train[:, 25] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 25])

for i in range(0, len(dat_str)):
    str_data_test[:, 25] = np.where(str_data_test[:, 25] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 25])

# 27
np.array(np.unique(str_data_train[:, 26], return_counts = True))
np.array(np.unique(str_data_test[:, 26], return_counts = True))
dat_str = np.array(['Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 26] = np.where(str_data_train[:, 26] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 26])

for i in range(0, len(dat_str)):
    str_data_test[:, 26] = np.where(str_data_test[:, 26] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 26])

# 28
np.array(np.unique(str_data_train[:, 27], return_counts = True))
np.array(np.unique(str_data_test[:, 27], return_counts = True))
dat_str = np.array(['Ex', 'Fa', 'Gd', 'Po', 'TA'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 27] = np.where(str_data_train[:, 27] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 27])

for i in range(0, len(dat_str)):
    str_data_test[:, 27] = np.where(str_data_test[:, 27] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 27])

# 29
np.array(np.unique(str_data_train[:, 28], return_counts = True))
np.array(np.unique(str_data_test[:, 28], return_counts = True))

str_data_train[:, 28] = np.where(str_data_train[:, 28] == 'N',
                                0, 1)

str_data_test[:, 28] = np.where(str_data_test[:, 28] == 'Y',
                                1, 0)

# 30
np.array(np.unique(str_data_train[:, 29], return_counts = True))
np.array(np.unique(str_data_test[:, 29], return_counts = True))
dat_str = np.array(['FuseA', 'FuseF', 'FuseP', 'Mix', 'NA', 'SBrkr'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 29] = np.where(str_data_train[:, 29] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 29])

for i in range(0, len(dat_str)):
    str_data_test[:, 29] = np.where(str_data_test[:, 29] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 29])

# 31
np.array(np.unique(str_data_train[:, 30], return_counts = True))
np.array(np.unique(str_data_test[:, 30], return_counts = True))
dat_str = np.array(['Ex', 'Fa', 'Gd', 'NA', 'TA'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 30] = np.where(str_data_train[:, 30] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 30])

for i in range(0, len(dat_str)):
    str_data_test[:, 30] = np.where(str_data_test[:, 30] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 30])

# 32
np.array(np.unique(str_data_train[:, 31], return_counts = True))
np.array(np.unique(str_data_test[:, 31], return_counts = True))
dat_str = np.array(['Maj1', 'Maj2', 'Min1', 'Min2', 'Mod', 'NA', 'Sev', 'Typ'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 31] = np.where(str_data_train[:, 31] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 31])

for i in range(0, len(dat_str)):
    str_data_test[:, 31] = np.where(str_data_test[:, 31] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 31])

# 33
np.array(np.unique(str_data_train[:, 32], return_counts = True))
np.array(np.unique(str_data_test[:, 32], return_counts = True))
dat_str = np.array(['Ex', 'Fa', 'Gd', 'NA', 'Po', 'TA'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 32] = np.where(str_data_train[:, 32] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 32])

for i in range(0, len(dat_str)):
    str_data_test[:, 32] = np.where(str_data_test[:, 32] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 32])

# 34
np.array(np.unique(str_data_train[:, 33], return_counts = True))
np.array(np.unique(str_data_test[:, 33], return_counts = True))
dat_str = np.array(['2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'NA'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 33] = np.where(str_data_train[:, 33] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 33])

for i in range(0, len(dat_str)):
    str_data_test[:, 33] = np.where(str_data_test[:, 33] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 33])

# 35
np.array(np.unique(str_data_train[:, 34], return_counts = True))
np.array(np.unique(str_data_test[:, 34], return_counts = True))
dat_str = np.array(['Fin', 'NA', 'RFn', 'Unf'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 34] = np.where(str_data_train[:, 34] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 34])

for i in range(0, len(dat_str)):
    str_data_test[:, 34] = np.where(str_data_test[:, 34] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 34])

# 36
np.array(np.unique(str_data_train[:, 35], return_counts = True))
np.array(np.unique(str_data_test[:, 35], return_counts = True))
dat_str = np.array(['Ex', 'Fa', 'Gd', 'NA', 'Po', 'TA'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 35] = np.where(str_data_train[:, 35] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 35])

for i in range(0, len(dat_str)):
    str_data_test[:, 35] = np.where(str_data_test[:, 35] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 35])

# 37
np.array(np.unique(str_data_train[:, 36], return_counts = True))
np.array(np.unique(str_data_test[:, 36], return_counts = True))
dat_str = np.array(['Ex', 'Fa', 'Gd', 'NA', 'Po', 'TA'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 36] = np.where(str_data_train[:, 36] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 36])

for i in range(0, len(dat_str)):
    str_data_test[:, 36] = np.where(str_data_test[:, 36] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 36])

# 38
np.array(np.unique(str_data_train[:, 37], return_counts = True))
np.array(np.unique(str_data_test[:, 37], return_counts = True))
dat_str = np.array(['N', 'P', 'Y'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 37] = np.where(str_data_train[:, 37] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 37])

for i in range(0, len(dat_str)):
    str_data_test[:, 37] = np.where(str_data_test[:, 37] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 37])

# 39
np.array(np.unique(str_data_train[:, 38], return_counts = True))
np.array(np.unique(str_data_test[:, 38], return_counts = True))
dat_str = np.array(['Ex', 'Fa', 'Gd', 'NA'])
        
for i in range(0, len(dat_str)):
    str_data_train[:, 38] = np.where(str_data_train[:, 38] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 38])

for i in range(0, len(dat_str)):
    str_data_test[:, 38] = np.where(str_data_test[:, 38] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 38])

# 40 
np.array(np.unique(str_data_train[:, 39], return_counts = True))
np.array(np.unique(str_data_test[:, 39], return_counts = True))
dat_str = np.array(['GdPrv', 'GdWo', 'MnPrv', 'MnWw', 'NA'])

for i in range(0, len(dat_str)):
    str_data_train[:, 39] = np.where(str_data_train[:, 39] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 39])

for i in range(0, len(dat_str)):
    str_data_test[:, 39] = np.where(str_data_test[:, 39] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 39])

# 41
np.array(np.unique(str_data_train[:, 40], return_counts = True))
np.array(np.unique(str_data_test[:, 40], return_counts = True))
dat_str = np.array(['Gar2', 'NA', 'Othr', 'Shed', 'TenC'])

for i in range(0, len(dat_str)):
    str_data_train[:, 40] = np.where(str_data_train[:, 40] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 40])

for i in range(0, len(dat_str)):
    str_data_test[:, 40] = np.where(str_data_test[:, 40] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 40])

# 42
np.array(np.unique(str_data_train[:, 41], return_counts = True))
np.array(np.unique(str_data_test[:, 41], return_counts = True))
dat_str = np.array(['COD', 'CWD', 'Con', 'ConLD', 'ConLI', 'ConLw', 'NA', 'New', 'Oth', 'WD'])

for i in range(0, len(dat_str)):
    str_data_train[:, 41] = np.where(str_data_train[:, 41] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 41])

for i in range(0, len(dat_str)):
    str_data_test[:, 41] = np.where(str_data_test[:, 41] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 41])

# 43
np.array(np.unique(str_data_train[:, 42], return_counts = True))
np.array(np.unique(str_data_test[:, 42], return_counts = True))
dat_str = np.array(['Abnorml', 'AdjLand', 'Alloca', 'Family', 'Normal', 'Partial'])

for i in range(0, len(dat_str)):
    str_data_train[:, 42] = np.where(str_data_train[:, 42] == dat_str[i], 
                                       i + 1, 
                                       str_data_train[:, 42])

for i in range(0, len(dat_str)):
    str_data_test[:, 42] = np.where(str_data_test[:, 42] == dat_str[i],
                                       i + 1, 
                                       str_data_test[:, 42])

str_data_train = str_data_train.astype(np.int_)
str_data_test = str_data_test.astype(np.int_)

#===================================================================================================================
#datos numericos
#===================================================================================================================

# 1
np.isnan(num_data_test).sum()
np.isnan(num_data_train).sum()

for i in range(0, num_data_train.shape[1]):
    num_data_train[:, i] = np.where(np.isnan(num_data_train[:, i]),
                                      temporary_stats[1, nums_cols[i]],
                                      num_data_train[:, i])

for i in range(0, num_data_test.shape[1]):
    num_data_test[:, i] = np.where(np.isnan(num_data_test[:, i]),
                                      temporary_mean_test[nums_cols_test[i]],
                                      num_data_test[:, i])

# Vamos a eliminar el id por que no trae info, la guardamos en otra variable
names_num_train = names_num_train[1:]
names_num_test = names_num_test[1:]

id_test = np.copy(num_data_test[:, 0])
id_train = np.copy(num_data_train[:, 0])

num_data_train = np.delete(num_data_train, 0, axis = 1)
num_data_test = np.delete(num_data_test, 0, axis = 1)

x_train = np.hstack((num_data_train, str_data_train))

x_test = np.hstack((num_data_test, str_data_test))

y_train = x_train[:, -1]
x_train = np.delete(x_train, 0, axis = 1)

mean = x_train.mean(axis = 0)
x_train -= mean
std = x_train.std(axis = 0)
x_train /= std
x_test -=mean
x_test /= std 
indices = np.random.permutation(id_train.shape[0])

x_train = x_train[indices]
y_train = y_train[indices]

# num_val_samples = int(x_train.shape[0]*0.3)

# x_val = x_train[:num_val_samples]
# y_val = y_train[:num_val_samples]

# x_train = x_train[num_val_samples:]
# y_train = y_train[num_val_samples:]
#===================================================================================================================
# MODELO
#===================================================================================================================

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

# model = keras.Sequential([
#     layers.Dense(320, activation = 'relu'),
#     layers.Dense(160, activation = 'relu'),
#     layers.Dense(160, activation = 'relu'),
#     layers.Dense(1)])

# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor = 'val_loss',
#     mode = 'auto',    
#     min_delta = 0,
#     patience = 15,
#     verbose = 1, 
#     restore_best_weights = True)

# model.compile(optimizer = 'rmsprop',
#               loss = 'mse',
#               metrics = ['mae'])

# history = model.fit(x_train,
#                     y_train,
#                     epochs = 200,
#                     batch_size = 16,
#                     callbacks = [early_stopping],
#                     validation_data = (x_val, y_val))

# history_dict = history.history
# history_dict.keys()

# import matplotlib.pyplot as plt
# history_dict = history.history
# loss_values = history_dict['loss']
# val_loss_values = history_dict['val_loss']
# epochs = range(1, len(loss_values)+1)
# plt.plot(epochs, loss_values, 'b', label = 'Training loss')
# plt.plot(epochs, val_loss_values, 'r', label = 'Validation loss')
# plt.title('Training and validations loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# plt.clf()                                                        # 1 
# acc_values = history_dict['mae']
# val_acc_values = history_dict['val_mae']
# plt.plot(epochs, acc_values, 'b', label = 'Training MAE')
# plt.plot(epochs, val_acc_values, 'r', label = 'Validation MAE')
# plt.title('Training and validations MAE')
# plt.xlabel('Epochs')
# plt.ylabel('MAE')
# plt.legend()
# plt.show()

#===================================================================================================================
# K-fold
#===================================================================================================================

def build_model():
    model = keras.Sequential([
        layers.Dense(160, activation = 'relu'),
        layers.Dense(80, activation = 'relu'),
        layers.Dense(1)])
    
    model.compile(optimizer = keras.optimizers.RMSprop(
        learning_rate = 0.0005), 
        loss = keras.losses.MeanAbsolutePercentageError(),
        metrics = ['mae'])
    
    return model

k = 5
num_val_samples = len(x_train) // k
num_epochs = 120
all_acc_histories = []

for i in range(k):
    print(f'Processing fold #{i}')
    val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]
    
    partial_x_train = np.concatenate(
        [x_train[: i * num_val_samples],
         x_train[(i + 1) * num_val_samples :]],
        axis = 0)
    
    partial_y_train = np.concatenate(
        [y_train[: i * num_val_samples],
         y_train[(i + 1) * num_val_samples :]],
        axis = 0)
    
    model = build_model()
            
    history = model.fit(partial_x_train, partial_y_train, 
              epochs = num_epochs, batch_size  = 16, verbose = 0)

    acc_history = history.history['mae']
    all_acc_histories.append(acc_history)


import matplotlib.pyplot as plt

average_acc_history = [
    np.mean([x[i] for x in all_acc_histories]) for i in range(num_epochs)]

plt.plot(range(1, len(average_acc_history) + 1), average_acc_history)
plt.xlabel('Epochs')
plt.ylabel('Validation accuracy')
plt.show()

model.evaluate(x_train, y_train)

summit= model(x_test)
summit.shape
summit = np.reshape(summit, (1459,))

import pandas as pd

submission = pd.DataFrame({"Id": id_test,"SalePrice": summit})
submission
submission['Id'] = submission['Id'].astype('int32')
submission.to_csv('C:\\Users\\Christian\\Desktop\\submission.csv', index=False)
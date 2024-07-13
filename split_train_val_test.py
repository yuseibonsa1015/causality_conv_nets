# -*- coding: utf-8 -*-

import pandas as pd
import os
import sklearn
import numpy as np 
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

seed=42

def get_or_create_datasetsCSVpaths(EXPERIMENT, CONDITIONING_FEATURE, csv_path, testset_size=0.2, validset_size=0.15):

    if EXPERIMENT=="prostate": #PICAI dataset, EXPERIMENTがprostate(前立腺)の場合
        path_tr = os.path.join(os.getcwd(),"dataset_PICAI", "csv_files",f"d_train_{CONDITIONING_FEATURE}_unfolded.csv")
        path_va = os.path.join(os.getcwd(),"dataset_PICAI","csv_files",f"d_val_{CONDITIONING_FEATURE}_unfolded.csv")
        path_te = os.path.join(os.getcwd(),"dataset_PICAI","csv_files",f"d_test_{CONDITIONING_FEATURE}_unfolded.csv") # 任意のcsvファイル（あるかどうかにかかわら字）までのパスを作成
        if not (os.path.exists(path_tr) and os.path.exists(path_va) and os.path.exists(path_te)): # いずれかのパス先のcsvファイルが存在しない場合

            df = pd.read_csv(csv_path) # 指定されたcsvファイルを読み込み、pandasのDataFrameとして格納
            #%% Split into training+validation and test 学習データ（訓練データ＋検証データ）とテストデータに分ける
            study1 = df.study_id # DataFrame「df」からstudy_id列を取り出してstudy1に格納
            #labels1 = df.label
            patients1 = df.patient_id # DataFrame「df」からpatient_id列を取り出してpatients1に格納
            gs = GroupShuffleSplit(n_splits=2, test_size=testset_size, random_state=seed) # 指定した変数の値に基づいてデータを分割するsklearnのクラスをgfとしてインスタンス化、n_splitsは分割回数（n_splits=2ならtrainとtestデータ分割するため1回の分割）、test_sizeはテストセットのサイズ指定（test_size=0.2なら全データの20％がテストデータになる)、rondom_stateシャッフルするランダムのシード
            trainval_idx, test_idx = next(gs.split(study1, groups=patients1)) # gsを使ってデータを分割、groupに基づいてstudy1も分割、nest()で分割したデータのインデックスを取得、格納, split()だけでは(array([0, 1, 4, 5, 6, 7], dtype=int64), array([2, 3], dtype=int64))この値になる。インデックスが欲しいので一番最初の要素をnext()で取得
            trainvalset = df.loc[trainval_idx] # インデックスに対応した値を抽出、格納
            testset = df.loc[test_idx] # インデックスに対応した値を抽出、格納
            #%% Split into training and validation
            study2 = trainvalset.study_id # 訓練データ内のstudy_id列の値をstudy2に格納
            #labels2 = trainvalset.label
            patients2 = trainvalset.patient_id # 訓練データ内のpatient_id列の値をpatients2に格納
            gs2 = GroupShuffleSplit(n_splits=2, test_size=validset_size, random_state=seed) # 指定した変数の値に基づいてデータを分割するsklearnのクラスをgf2としてインスタンス化、学習データからさらに訓練データと検証データ分割するため
            train_idx, val_idx = next(gs2.split(study2, groups=patients2)) # gs2を使ってデータを分割
            trainset = trainvalset.reset_index().loc[train_idx] # 訓練データを抽出
            valset = trainvalset.reset_index().loc[val_idx] # 検証データを抽出
            #%%    
            # # shuffle the DataFrame rows
            trainset = trainset.sample(frac = 1, random_state=seed) 
            valset = valset.sample(frac = 1, random_state=seed)
            testset = testset.sample(frac = 1, random_state=seed) # データをシャッフル、frac=１でデータ全体をシャッフル対象にする、rondom_stateはシード値
            # Save
            trainset.to_csv(path_tr,index=False)
            valset.to_csv(path_va,index=False)
            testset.to_csv(path_te,index=False) # 作成したデータを、あらかじめ作成したファイルに保存
            print("get_or_create_datasetsCSVpaths(): created the three CSV files") # ログ表示：get_or_create_datasetsCSVpaths()：3つのCSVファイルを作成
            
        else:
            print("get_or_create_datasetsCSVpaths(): the three CSV files have been already created before") # ログ表示：get_or_create_datasetsCSVpaths()：3つのCSVファイルは事前に作成済み
        
    elif EXPERIMENT == "procancer":
        
        desired_test_ratio=0.20
        desired_val_ratio=0.15

        path_tr = os.path.join(os.getcwd(),"dataset_procancer", "csv_files",f"d_tr_{EXPERIMENT}_{CONDITIONING_FEATURE}.csv")
        path_va = os.path.join(os.getcwd(),"dataset_procancer","csv_files",f"d_va_{EXPERIMENT}_{CONDITIONING_FEATURE}.csv")
        path_te = os.path.join(os.getcwd(),"dataset_procancer","csv_files",f"d_te_{EXPERIMENT}_{CONDITIONING_FEATURE}.csv") # 任意のcsvファイルまでのパスを作成

        if not (os.path.exists(path_tr) and os.path.exists(path_va) and os.path.exists(path_te)): # 指定したパス先のいずれかのファイルが存在しない場合

            df = pd.read_csv(csv_path) # csvファイルを読み込んでデータフレームに格納
            #%% Split into training+validation and test
            data1 = df.data_index
            labels1 = df.groundtruth
            patients1 = df.patient_id # dfのpatient_id列の値をpetients1に格納

            # gs = GroupShuffleSplit(n_splits=2, test_size=testset_size, random_state=seed)
            # trainval_idx, test_idx = next(gs.split(series1, groups=patients1))            
            cv = StratifiedGroupKFold(n_splits=int(1/desired_test_ratio), shuffle=True) # K個のデータに分割(一つはテスト用、残りのk-1個は学習用に使う)するクラスをcvとしてインスタンス化、n_splitsは分割するデータの数、shuffleはデータを分割する前にシャッフルするかどうか
            trainval_idx, test_idx = next(cv.split(data1, labels1, patients1)) # cvで学習用とテスト用に分割した後のデータのindexを格納
            
            trainvalset = df.loc[trainval_idx]
            testset = df.loc[test_idx] # indexに対応した値を抽出
            #%% Split into training and validation
            data2 = trainvalset.data_index
            labels2 = trainvalset.groundtruth
            patients2 = trainvalset.patient_id # 指定したラベル名に対応する値を抽出、格納

            # gs2 = GroupShuffleSplit(n_splits=2, test_size=validset_size, random_state=seed)
            # train_idx, val_idx = next(gs2.split(series2, groups=patients2))
            cv = StratifiedGroupKFold(n_splits=int(1/desired_val_ratio), shuffle=True) # cvを再度インスタンス化（今回は設定内容が一緒なため意味がない）
            train_idx, val_idx = next(cv.split(data2, labels2, patients2)) # cvで学習用からさらに訓練用と検証用に分割、そのindexを格納
            trainset = trainvalset.reset_index().loc[train_idx] # reset_indexは今のバラバラなindexを'index'列名として新しく追加して残しておき、新たに0からindexを振りなおす
            valset = trainvalset.reset_index().loc[val_idx] # indexに対応した値を抽出
            #%%    
            # # shuffle the DataFrame rows
            trainset = trainset.sample(frac = 1, random_state=seed)
            valset = valset.sample(frac = 1, random_state=seed)
            testset = testset.sample(frac = 1, random_state=seed) # データをシャッフル
            # Save
            trainset.to_csv(path_tr,index=False) # データを指定したパス先に保存
            print(f"Saved TRAINSET csv file, with proportion of events: {labels2.reset_index().loc[train_idx].mean()}") 

            valset.to_csv(path_va,index=False)
            print(f"Saved VALSET csv file, with proportion of events: {labels2.reset_index().loc[val_idx].mean()}")

            testset.to_csv(path_te,index=False)
            print(f"Saved TESTSET csv file, with proportion of events: {labels1.reset_index().loc[test_idx].mean()}") # ログ表示：TRAINSETのcsvファイルにイベントの比率を保存、train_idx列に対応する値の平均値も表示してデータの特性を表示させている
        else:
            print("get_or_create_datasetsCSVpaths(): the three CSV files have been already created before") # ログ表示：get_or_create_datasetsCSVpaths()：3つのCSVファイルは事前に作成済み。
    



    if EXPERIMENT=="breakhis": #breakhistopathology dataset
        desired_test_ratio=0.20
        desired_val_ratio=0.15

        path_tr = os.path.join(os.getcwd(),"dataset_breakhis", "csv_files",f"d_train_{CONDITIONING_FEATURE}_unfolded.csv")
        path_va = os.path.join(os.getcwd(),"dataset_breakhis","csv_files",f"d_val_{CONDITIONING_FEATURE}_unfolded.csv")
        path_te = os.path.join(os.getcwd(),"dataset_breakhis","csv_files",f"d_test_{CONDITIONING_FEATURE}_unfolded.csv") # 任意のcsvファイルまでのパスを作成
        if not (os.path.exists(path_tr) and os.path.exists(path_va) and os.path.exists(path_te)): # いずれかのパス先のcsvファイルが存在しない場合

            df = pd.read_csv(csv_path) # csv_path先のファイルを読み込み、dfにデータフレームとして格納
            
            #%% Split into training+validation and test
            # xs = df.image
            xs = df.index # dfのindexをxsに格納

            ys = df.binary_target # dfの'binary_target'列を抽出、ysに格納

            trainval_idx, test_idx, _, _,= sklearn.model_selection.train_test_split(xs, ys,
                                                    test_size=desired_test_ratio,
                                                    random_state=seed,
                                                    stratify=ys) # train_test_splitでデータ(index)を分割、xsはindex、ysは特徴量？、test_sizeはテストデータのサイズ、stratifyは分布を保持したままシャッフル（例えばysは0, 1のいバイナリだが、0が偽、1が正解のような性質を持つデータの時、0, 1が同じ割合で分割されるように設定している

            trainvalset = df.loc[trainval_idx] #temprorary 学習（訓練＋検証）データを格納
            testset = df.loc[test_idx] #final テストデータを格納

            #%% Split into training and validation
            # xs = trainvalset.image
            xs = trainvalset.index # 学習（訓練＋検証）データのindexを格納

            ys = trainvalset.binary_target # 学習（訓練＋検証）データの'binary_target'列を抽出、格納
            train_idx, val_idx, _, _ = sklearn.model_selection.train_test_split(xs, ys,
                                                    test_size=desired_val_ratio, #real validation
                                                    random_state=seed,
                                                    stratify=ys) # 学習（訓練＋検証）データからさらに訓練データと検証データに分割

            trainset = trainvalset.loc[train_idx] #final 訓練データを格納
            valset = trainvalset.loc[val_idx] #final 検証データを格納
            # trainset = trainvalset.reset_index().loc[train_idx] #final
            # valset = trainvalset.reset_index().loc[val_idx] #final
            #%%    
            # # shuffle the DataFrame rows
            trainset = trainset.sample(frac = 1, random_state=seed) 
            valset = valset.sample(frac = 1, random_state=seed)
            testset = testset.sample(frac = 1, random_state=seed)# 分割後のデータをシャッフル
            # Save
            trainset.to_csv(path_tr,index=False)
            valset.to_csv(path_va,index=False)
            testset.to_csv(path_te,index=False) # 分割後シャッフルしたデータをcsvファイルに保存
            print("get_or_create_datasetsCSVpaths(): created the three CSV files") # ログ表示：get_or_create_datasetsCSVpaths()：3つのCSVファイルを作成。
            
        else:
            print("get_or_create_datasetsCSVpaths(): the three CSV files have been already created before") # ログ表示：get_or_create_datasetsCSVpaths()：3つのCSVファイルは事前に作成済み。
        


    return path_tr, path_va, path_te # データファイルのパスを返す

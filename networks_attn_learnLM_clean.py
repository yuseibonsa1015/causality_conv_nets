import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import time
import random

## Info (november 2023)
# The name of this script indicates the possibility to insert the vision attention mechanism (_attn_)
# and the learning of the Lehmer Mean power (_learnLM_). However, these are improvements for future work,
# and therefore they are here not completely coded nor tested. Please use the settings as in the main published paper.

######
class STEFunction(torch.autograd.Function): # Mulcat手法に使う
    '''
    https://discuss.pytorch.org/t/binary-activation-function-with-pytorch/56674/4
    '''
    @staticmethod
    def forward(ctx, input): # 順伝搬を実行, 'ctx'はコンテキストオブジェクト（バックプロパゲーション中に必要な情報を保存するために使用？）, 'input'はテンソルで、順伝搬の入力
        return (input > 0).float() # 出力はinputが0を超えるかどうかでバイナリ出力
    @staticmethod
    def backward(ctx, grad_output): # 逆伝搬を実行, 'grad_output'は次のレイヤから伝搬される勾配
        return F.hardtanh(grad_output) #出力はhardtanh関数で勾配を-1から1の間に制限
class StraightThroughEstimator(nn.Module): # 親クラスがnn.Module, Mulcat手法に使う
    '''
    https://discuss.pytorch.org/t/binary-activation-function-with-pytorch/56674/4
    '''
    def __init__(self):
        super(StraightThroughEstimator, self).__init__() # super(親クラスのオブジェクト, self).親クラスのメソッド　で親クラスで定義したメソッドを継承して使える
    def forward(self, x): # 順伝搬
        x = STEFunction.apply(x) # 入力ｘに対してSTEFunction.applyメソッドを呼び出してバイナリ活性化とストレートスルー推定を適用, バイナリ活性化関数の勾配消失問題を回避するため？
        return x

##############################################
class CausalityMapBlock(nn.Module): # 因果マップを計算、出力するクラス, forward()で特徴マップxを入力し因果マップcausality_mapsを出力
    def __init__(self, elems, causality_method, fixed_lehmer_param=None):
        '''
        elems: (int) square of the number of elements in each featuremap at the conv bottleneck, (eg: F=3x3=9-->81)
        causality_method: (str) max or lehmer
        fixed_lehmer_param: (float) power of the Lehmer Mean, such as -2.0, 0.0, or 1.0, etc.
            When it is not passed - it is None - this makes the lehmer_seed learnable by backprop with torch.nn.Parameter...(to be continued)
        '''
        super(CausalityMapBlock, self).__init__() # 親クラスの初期化メソッドを呼び出す
        self.elems = elems # 引数をクラスのインスタンス変数として保存, 特徴マップのチャネル数？
        self.causality_method =causality_method # 引数をクラスのインスタンス変数として保存
        if self.causality_method=="lehmer": # レーマー法で因果的特徴抽出を行う場合
            if fixed_lehmer_param is not None: # 固定されたレーマーパラメータを引数として渡している場合
                self.lehmer_seed=float(fixed_lehmer_param) # fixed_lehmer_paramをfloat型に変換してself.lehmer_seedに格納
            else: # 渡していない場合
                self.lehmer_seed=torch.nn.Parameter(torch.tensor([0.0],device="cuda")) # 0.0を含むテンソルを生成し、device="cuda"でcuda上に配置, torch.nn.Parameter()は学習可能なパラメータを生成する関数
            print(f"INIT - CausalityMapBlock: LEHMER with self.lehmer_seed={self.lehmer_seed}") # レーマー法で使うパラメータを初期化完了したことを表示
        else: #"max"
            print(f"INIT - CausalityMapBlock self.causality_method: {self.causality_method}") # 因果的特徴抽出がMax法であることを表示

    def forward(self,x): #(bs,k,n,n), 順伝搬
        
        if torch.isnan(x).any(): # テンソル'x'にNaN値が含まれている場合, torch.isnan(x)はxの各要素にNaN値があるかどうかチェックし、同じ形状のBool値を返す（NaN値をTrue、そのほかをFalse）, .any()はプールテンソル全体に一つでもTrueがあれば全体としてTrueを返す
            print(f"...the current feature maps object contains NaN") # 特徴マップにNaN値が含まれていることを表示
            raise ValueError # エラーを発生させる
        maximum_values = torch.max(torch.flatten(x,2), dim=2)[0] # flatten: (bs,k,n*n)、bsはバッチサイズ, (bs, k, n, n)を2次元目（3つ目）から平滑化して(bs,k,n*n)、2次元目（3つ目）にそって最大値取得（n*n行列をを一列に平滑化した列の最大値） max: (bs,k), torch.flatten(x,2)によりxを2次元目（3つ目）から平滑化, torch.max(テンソル, dim=2)[0]によりテンソルを2次元目（3つ目の次元）にそって最大値を取得、[0]は最大値、[1]はそのindexを返す, 特徴マップは
        MAX_F = torch.max(maximum_values, dim=1)[0]  #MAX: (bs,), maximum_valuesの1次元（２つめ）に沿って最大値を取得, 一つのバッチ内の最大値を取得
        x_div_max=x/(MAX_F.unsqueeze(1).unsqueeze(2).unsqueeze(3) + 1e-8) #TODO added epsilon; #implement batch-division: each element of each feature map gets divided by the respective MAX_F of that batch, xの要素をバッチごとの最大値で割ることによって正規化, 1e-8はゼロ除算を回避
        x = torch.nan_to_num(x_div_max, nan = 0.0) # nan_to_num()により計算の過程で発生するNaN値を0に変換

        ## After having normalized the feature maps, comes the distinction between the method by which computing causality.
        #Note: to prevent ill posed divisions and operations, we sometimes add small epsilon (e.g., 1e-8) and nan_to_num() command.
        if self.causality_method == "max": #Option 1 : max values, Max法で因果的特徴抽出を行う場合

            sum_values = torch.sum(torch.flatten(x,2), dim=2) # それぞれのn*n行列を一列に平滑化し、その一列の要素の合計をsum_valuesに格納
            if torch.sum(torch.isnan(sum_values))>0: # is_nan()によりsum_valuesの要素の中のNaN値をTrue、そのほかをFalseとし、sum_valuesと同じ形状のテンソルで返し、そのテンソルのTrue（1）の合計が0より大きい場合、中にNaN値が含まれていることになるので、その場合
                sum_values = torch.nan_to_num(sum_values,nan=0.0)#sostituire gli eventuali nan con degli zeri, NaN値を0.0に変換
            
            maximum_values = torch.max(torch.flatten(x,2), dim=2)[0] # maximum_valuesの1次元（２つめ）に沿って最大値を取得, 一つのバッチ内の最大値を取得
            mtrx = torch.einsum('bi,bj->bij',maximum_values,maximum_values) #batch-wise outer product, the max value of mtrx object is 1.0, ２つの行列（今回は同じ行列）の要素の順序を考慮したすべての要素同士の積を要素に持つ行列（k×ｋ）を格納, torch.einsum()はサブテキストの指定に応じて渡したテンソルを計算する, 
            tmp = mtrx/(sum_values.unsqueeze(1) +1e-8) #TODO added epsilon, 特徴マップごとの要素の和を次元を調整、ゼロ除算を回避するための調整をしてmtrxに割ることによってMax法による因果マップを計算
            causality_maps = torch.nan_to_num(tmp, nan = 0.0) # NaN値を0.0に変換

        elif self.causality_method == "lehmer": #Option 2 : Lehmer mean   
            
            x = torch.flatten(x,2) # [b,k,n*n], eg [16,512,8*8]
            #compute the outer product of all the pairs of flattened featuremaps.
            # This provides the numerator (without lehemer mean, yet) for each cell of the final causality maps:
            cross_matrix = torch.einsum('bmi,bnj->bmnij', x, x) #eg, [16,512,512,64,64]  symmetric values 
            cross_matrix = cross_matrix.flatten(3) #eg, [16,512,512,4096]

            # apply lehmer mean function to each flattened cell (ie, vector) of the kxk matrix:
            # first, compute the two powers of the cross matrix
            p_plus_1_powers = torch.nan_to_num(torch.pow(cross_matrix+1e-8, self.lehmer_seed+1)) #eg, [16,512,512,4096]
            
            p_powers = torch.nan_to_num(torch.pow(cross_matrix+1e-8, self.lehmer_seed)) #eg, [16,512,512,4096]

            numerators = torch.nan_to_num(torch.sum(p_plus_1_powers, dim=3)) + 1e-8   
            denominators = torch.nan_to_num(torch.sum(p_powers, dim=3)) + 1e-8  #eg, [16,512,512]
            lehmer_numerators = torch.nan_to_num(torch.div(numerators,denominators)) + 1e-8    #[bs,k,k]
            #############            
            # then the lehmer denominator of the causality map:
            # it is the lehemr mean of the single feature map, for all the feature maps by column

            p_plus_1_powers_den = torch.nan_to_num(torch.pow(x.abs()+1e-8, self.lehmer_seed+1))

            p_powers_den = torch.nan_to_num(torch.pow(x.abs()+1e-8, self.lehmer_seed)) ##TODO

            numerators_den = torch.nan_to_num(torch.sum(p_plus_1_powers_den, dim=2)) + 1e-8   
            denominators_den = torch.nan_to_num(torch.sum(p_powers_den, dim=2)) + 1e-8   
            lehmer_denominator = torch.nan_to_num(torch.div(numerators_den,denominators_den), nan=0) + 1e-8    
            #and finally obtain the causality map values by computing the division
            causality_maps = torch.nan_to_num(torch.div(lehmer_numerators, lehmer_denominator.unsqueeze(1)), nan=0)
            
        else:
            print(self.causality_method) # we implemented only MAX and LEHMER options, so every other case is a typo/error
            raise NotImplementedError

        # print(causality_maps)    
        return causality_maps #因果マップを出力


class CausalityFactorsExtractor(nn.Module): # Mulcat手法の実装
    def __init__(self, causality_direction, causality_setting):
        
        super(CausalityFactorsExtractor, self).__init__()
        self.causality_direction = causality_direction #eg, "causes" or "effects"
        self.causality_setting = causality_setting #eg, "mulcat,mulcatbool,mul,mulbool"
        self.STE = StraightThroughEstimator() #Bengio et al 2013
        self.relu = nn.ReLU()
        print(f"INIT - CausalityFactorsExtractor: self.causality_direction={self.causality_direction}, self.causality_setting={self.causality_setting}")

    def forward(self, x, causality_maps):
        '''
        x [bs, k, h, w]: the feature maps from the original (regular CNN) branch;
        causality_maps [bs, k, k]: the output of a CausalityMapsBlock().

        By leveraging algaebric transformations and torch functions, we efficiently extracts the causality factors with few lines of code.
        '''
        triu = torch.triu(causality_maps, 1) #upper triangular matrx (excluding the principal diagonal)
        tril = torch.tril(causality_maps, -1).permute((0,2,1)).contiguous() #lower triangular matrx  (excluding the principal diagonal), transposed for easier elementwise computation with the upper

        e = tril - triu
        e = self.STE(e)
        e = e.permute((0,2,1))

        f = triu - tril
        f = self.STE(f)
        bool_matrix = e + f #sum of booleans is the OR logic

        by_col = torch.sum(bool_matrix, 2)
        by_row = torch.sum(bool_matrix, 1)

        if self.causality_direction=="causes":
            if self.causality_setting == "mulcat" or self.causality_setting == "mul":
                causes_mul_factors = by_col - by_row # the factor of a featuremap is how many times it causes some other featuremap minus how many times it is caused by other feature maps
                causes_mul_factors=self.relu(causes_mul_factors)

            elif self.causality_setting == "mulcatbool" or self.causality_setting == "mulbool":
                # causes_mul_factors = 1.0*((by_col - by_row)>0) # the factor of a featuremap is 1 (pass) iff it causes some other featuremap more than how many times itself is caused by other feature maps, 0 (not pass) otherwise
                causes_mul_factors = self.STE(by_col - by_row) # TODO 30 oct: differentiable version for torch autograd

        elif self.causality_direction=="effects":
            if self.causality_setting == "mulcat" or self.causality_setting == "mul":
                causes_mul_factors = by_row - by_col # 
                causes_mul_factors=self.relu(causes_mul_factors)

            elif self.causality_setting == "mulcatbool" or self.causality_setting == "mulbool":
                # causes_mul_factors = 1.0*((by_row - by_col)>0) # 
                causes_mul_factors = self.STE(by_row - by_col) # # TODO 30 oct: differentiable version for torch autograd
        
        else:
            print("Personal error: unrecognised self.causality_direction")
            raise ValueError        

        ## Directly return the "attended" ("causally"-weighted) version of x
        return torch.einsum('bkmn,bk->bkmn', x, causes_mul_factors)#multiply each (rectified) factor for the corresponding 2D feature map, for every minibatch

class Identity(nn.Module): # 入力をそのまま出力する関数を定義しているクラス
    def __init__(self):
        super(Identity, self).__init__() # nn.Moduleの__init__を継承
    def forward(self, x): 
        return x # 入力をそのまま出力する

# use this version, whose code have been cleaned, removed unnecessary parts, and improved globally.
class Resnet18CA_clean(nn.Module):
    def __init__(self, dim, channels, num_classes, is_pretrained, is_feature_extractor, causality_aware=False, causality_method="max", LEHMER_PARAM=None, causality_setting="cat", visual_attention=False, MULCAT_CAUSES_OR_EFFECTS="causes"):
            super(Resnet18CA_clean, self).__init__()
            self.img_size = dim # 入力画像のサイズ
            self.channels = channels # チャネル数
            self.num_classes = num_classes # 分類クラスの数
            self.is_pretrained = is_pretrained # 事前学習済みモデルを使用するかどうか
            self.is_feature_extractor = is_feature_extractor # feature_extractorとして使用するかどうか, [True, False]
            
            self.causality_aware = causality_aware # 画像内の因果関係を考慮した分類を行うかどうか, [True, False]
            self.causality_method = causality_method # 画像内の因果的信号をどのように計算するか, ["max", "lehmer"]
            self.causality_setting = causality_setting # 手法の選択, ["cat", "mulcat", "mul"]

            if LEHMER_PARAM is not None: # レーマー法のパラメータを引数に渡した場合
                self.LEHMER_PARAM = LEHMER_PARAM # その引数をself.LEHMER_PARAMに格納

            # self.visual_attention = visual_attention #boolean
            self.MULCAT_CAUSES_OR_EFFECTS = MULCAT_CAUSES_OR_EFFECTS #TODO 21 luglio, 引数MULCAT_CAUSES_OR_EFFECTSをself.MULCAT_CAUSES_OR_EFFECTSに格納

            if self.is_pretrained: # 事前学習済みモデルを使用する場合
                print("is_pretrained=True-------->loading imagenet weights")
                model = resnet18(pretrained=True) # ImageNet（カラー画像のデータベース）で学習された重みを使用, modelに学習された重みをロードさせたResnet18を格納, この重みは一般的な画像分類タスクに対して良い初期値を提供するためモデルの収束が早くなる
            else: # 事前学習済みモデルを使用しない場合
                print("is_pretrained=False---------->init random weights")
                model = resnet18() # ランダムに初期化された重みをもつResnet18を作成

            if self.channels == 1: # チャネルが一つの場合, グレースケール画像のこと               
                model.conv1 = nn.Conv2d(1, 64,kernel_size=7, stride=2, padding=3, bias=False, device='cpu') #output_size = ((input_size - kernel_size + 2 * padding) / stride) + 1, ぐれーすけーるに対応した最初の畳み込み層を作成, 入力チャネル1、出力チャネル64(フィルタが64個）、カーネルサイズ7、ストライド2、パディング3、バイアス項なし、使用デバイス
            elif self.channels==3: # チャネルが3場合, カラー画像（RGB画像）のこと, カラー画像は赤（R）、緑（G）、青（B）の3舞の画像の重ね合わせで表現される, この3枚の画像をチャネルという
                if self.is_feature_extractor: # 特徴抽出器として使用する場合
                    for param in model.parameters(): #freeze the extraction layers
                        param.requires_grad = False # モデルのすべてのパラメータの'required_grad'を'False'に設定, モデルのパラメータが固定されて学習時に更新されなくなる
            
            self.starting_block = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool) #output size is halved # nn.Sequential()で複数の層を順番に通過するように設定, conv1(畳み込み層)、bn1(バッチ正則化層、層が深くなるについて勾配が焼失しうまく学習できなくなる問題を回避宇するために身にバッチ全体のデータを正則化する)
            # self.starting_block = nn.Sequential(model.conv1, model.bn1, nn.LeakyReLU(), model.maxpool) #output size is halved

            self.layer1 = model.layer1 #output size is halved, 作成したモデルの第１層を抽出し、self.layer1に格納, 出力サイズが半分になる？
            self.layer2 = model.layer2 #output size is halved
            self.layer3 = model.layer3 #output size is halved
            self.layer4 = model.layer4 #output size is halved

            model.avgpool = Identity() # Cancel adaptiveavgpool2d layer to get feature maps of size, say, 7x7, Resnet18では最終層の'AdaptiveAvgPool2d'レイヤにより最終的な512枚の特徴マップ（7×7）の各要素の平均をとって512枚の特徴マップが1×1になるが、それをIdentity()でキャンセルし、7×7の特徴マップが出力されるようにする,        
            model.fc = Identity() # Cancel classification layer to get only feature extractor, 分類を行うための全結合層model.fcをIdentityに置き換えることで、この層をスキップし、特徴抽出のみを行う
            self.ending_block = nn.Sequential(model.avgpool, model.fc) # 上記の二つの層をまとめて一つのブロックにする            

            self.last_ftrmap_size = int(self.img_size/(2**5)) #outputsize is the original one divided by 32., 畳み込み層とプーリング層を通過するとサイズが半分になる。その操作を5回繰り返すので、self.img_sizeを2の5乗（３２）を割った値をself.last_ftrmap_sizeに格納, 入力画像サイズが224×224の場合特徴マップのサイズは7×7となる
            self.last_ftrmap_number = 512 #TODO 512 for ResNet18, 最終的な特徴マップのチャネル数を512とする

            if self.causality_aware: # 因果関係を考慮する場合
                ## initialize the modules for causality-driven networks
                if LEHMER_PARAM is not None: # レーマーパラメータを引数に渡していた場合
                    self.causality_map_extractor = CausalityMapBlock(elems=self.last_ftrmap_number, causality_method=self.causality_method, fixed_lehmer_param = self.LEHMER_PARAM) # レーマーパラメータも引数として渡して因果マップ生成クラスをself.causality_map_extractorとしてインスタンス化
                else:
                    self.causality_map_extractor = CausalityMapBlock(elems=self.last_ftrmap_number, causality_method=self.causality_method) # レーマーパラメータを引数として渡さず因果マップ生成クラスをself.causality_map_extractorとしてインスタンス化

                self.causality_factors_extractor = CausalityFactorsExtractor(self.MULCAT_CAUSES_OR_EFFECTS, causality_setting) # Mulcat手法による因果的特徴抽出クラスをself.causality_factors_extractorとしてインスタンス化
                ## and then set the classifier dimension, accordingly
                if self.causality_setting == "cat": #[1, n*n*k + k*k], Cat手法を用いる場合
                    self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size + self.last_ftrmap_number*self.last_ftrmap_number, self.num_classes) # 画像分類器の線形層を定義, nn.Linear(入力サイズ, 出力サイズ)はインスタンス化した後の引数xが（まず[1, 渡した入力サイズ]になるよう連結し？）重みとバイアスによるネットワークを介して[1, 渡した出力サイズ]形状になるように出力する, (特徴マップのチャネル数（k=512）*特徴マップのサイズ（n=7）*特徴マップのサイズ（n=7）, 分類クラス数)。これはResnet18の場合のパラメータ
                elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool"): #[1, 2*n*n*k], Mulcat手法を用いる場合
                    self.classifier = nn.Linear(2 * self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size, self.num_classes) # [1, 2*n*n*k]になるように特徴マップを連結
                elif self.causality_setting == "mul" or self.causality_setting == "mulbool": #TODO 18 settembre, どの手法かわからない
                    self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size, self.num_classes) # [1, n*n*k]になるように特徴マップを連結
           
            else: #regular, not causally driven network, 因果的特徴を考慮しない場合
                # if self.visual_attention: #TODO hardcoded, considerando attn2_4 e attn3_4, intanto solo per la versione non causale, poi preparare il codice anche per quela causale quindi mettere anche nel IF sopra
                #     self.classifier = nn.Linear(128*12*12 + 256*6*6 + 512*self.last_ftrmap_size*self.last_ftrmap_size, self.num_classes)
                # else:
                    self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size, self.num_classes) # [1, n*n*k]になるように特徴マップを連結


            # self.STE = StraightThroughEstimator() #Bengio et al 2013
            self.softmax = nn.Softmax(dim=1) #TODO 12 luglio, added, テンソルの値を0から1の確率に変換するSoftmax関数をインスタンス変数に格納, dim=1は1次元目(２つめ)に沿ってソフトマックスを適用

            # self.softmax = nn.LogSoftmax(dim=1) #TODO 26 ottobre

    def forward(self, x): # 入力：画像データx, 出力：特徴マップx, 因果マップcausality_map
        if torch.isnan(x).any(): # 入力テンソルxにNaN値がある場合, isnan(x)でxの要素をNaN値をTrue、そのほかをFalseに変換して同じ形状のまま返す, any()は要素の中のいずれか一つでもTrueがある場合、全体としてTrueを返す
            print("Personal error: FORWARD - the input was corrupted with NaN")
            raise ValueError # エラーを発生

        # print(f"forward - x: {x.size()}")
        x = self.starting_block(x) # xを設定したself.starting_blockに通す
        # print(f"forward - x strblk: {x.size()}")
        x_layer1 = self.layer1(x) # 
        # print(f"forward - x_layer1: {x_layer1.size()}")
        x_layer2 = self.layer2(x_layer1)
        # print(f"forward - x_layer2: {x_layer2.size()}")
        x_layer3 = self.layer3(x_layer2)
        # print(f"forward - x_layer3: {x_layer3.size()}")
        x_layer4 = self.layer4(x_layer3) # ResNet18にもともと定義されていた4層のレイヤに順番に通す
        # print(f"forward - x_layer4: {x_layer4.size()}")
        x = self.ending_block(x_layer4) # 設定した最終層に通す, 通常の最終層で行うavgpool（7×7×512を1×1×512に変換）、fc(全結合層1×分類クラス数)をキャンセルした層
        # print(f"forward - x ednblk: {x.size()}")
      
        if torch.isnan(x).any(): # NaN値がある場合
            print("Personal error: FORWARD - after passing through the feature extractor (conv net), x was corrupted with NaN")
            raise ValueError # エラーを発生させる
        
        if list(x.size()) != [int(x.size(0)), self.last_ftrmap_number, self.last_ftrmap_size, self.last_ftrmap_size]: # xの形状が期待された形状(k×n×n、ResNet18の場合512×7×7)でない場合
            x = torch.reshape(x, (int(x.size(0)), self.last_ftrmap_number, self.last_ftrmap_size, self.last_ftrmap_size)) # xを期待された形状(k×n×n、ResNet18の場合512×7×7)に再調整する

        causality_maps = None #initialized to none, # 因果マップをNoneで初期化

        if self.causality_aware:
            causality_maps = self.causality_map_extractor(x)       
                 
            if self.causality_setting == "cat": # 分類がcat手法の場合
                    x = torch.cat((torch.flatten(x, 1), torch.flatten(causality_maps, 1)), dim=1) # 特徴マップx、因果マップをバッチごとに一列に平滑化し横軸に沿って連結
            elif self.causality_setting == "mul" or self.causality_setting == "mulbool": # 分類がmul手法の場合
                    x_c = self.causality_factors_extractor(x, causality_maps)
                    x = torch.flatten(x_c, 1) #substitute the actual features with the filtered version of them.
            elif self.causality_setting == "mulcat" or self.causality_setting == "mulcatbool": #need to concatenate the x_c to the actual original features x, 画像分類の手法がmulcat手法の場合
                    x_c = self.causality_factors_extractor(x, causality_maps)
                    x = torch.cat((torch.flatten(x, 1), torch.flatten(x_c, 1)), dim=1) #cat the feature maps with their causality-enhanced version  
        
        else: #traditional, non causal:, 従来のCNNの場合
            x = torch.flatten(x, 1) # flatten all dimensions except batch, 特徴マップをバッチごとに一列に平滑化
   
        x = self.classifier(x) # 一列に平滑化された特徴テンソル（1×分類クラス数）に応じたスコア（1×クラス数）に重みとバイアスを介して線形変換
        x = self.softmax(x)  # ソフトマックス関数をかませることにより、値を確率に変換

        return x, causality_maps #return the logit of the classification, and the causality maps for optional visualization or some metric manipulation during training, 特徴マップxと因果マップcausality_mapを出力
    


















######## Old version, possibly with issues. Do not use. Utilize the cleaned version (above) instead.
# class Resnet18CA(nn.Module):
#     def __init__(self, dim, channels, num_classes, is_pretrained, is_feature_extractor, causality_aware=False, causality_method="max", LEHMER_PARAM=None, causality_setting="cat", visual_attention=False, MULCAT_CAUSES_OR_EFFECTS="causes"):
#             super(Resnet18CA, self).__init__()
#             self.img_size = dim
#             self.channels = channels
#             self.num_classes = num_classes
#             self.is_pretrained = is_pretrained
#             self.is_feature_extractor = is_feature_extractor
            
#             self.causality_aware = causality_aware
#             self.causality_method = causality_method
#             # self.LEHMER_PARAM = LEHMER_PARAM
#             self.causality_setting = causality_setting #

#             self.visual_attention = visual_attention #boolean
#             self.MULCAT_CAUSES_OR_EFFECTS = MULCAT_CAUSES_OR_EFFECTS #TODO 21 luglio

#             if self.is_pretrained:
#                 model = resnet18(pretrained=True)
#             else:
#                 model = resnet18()

#             if self.channels == 1:
#                 model.conv1 = nn.Conv2d(1, 64,kernel_size=7, stride=2, padding=3, bias=False, device='cpu') #output_size = ((input_size - kernel_size + 2 * padding) / stride) + 1
#             elif self.channels==3:
#                 if self.is_feature_extractor:
#                     for param in model.parameters(): #freeze the extraction layers
#                         param.requires_grad = False
            
#             ## creating the structure of our custom model starting from the original building blocks of resnet:
#             # starting block, layer1, layer2, layer3, layer4, ending block, and classifier

#             self.starting_block = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool) #output size is halved
#             self.layer1 = model.layer1 #output size is halved
#             self.layer2 = model.layer2 #output size is halved
#             self.layer3 = model.layer3 #output size is halved
#             self.layer4 = model.layer4 #output size is halved
#             #here, outputsize is the original one divided by 32.

#             model.avgpool = Identity() # Cancel adaptiveavgpool2d layer to get feature maps of size, say, 7x7       
#             model.fc = Identity() # Cancel classification layer to get only feature extractor
#             self.ending_block = nn.Sequential(model.avgpool, model.fc)
            
#             ##self.features = model
#             if self.visual_attention:
#                 ## define the attention blocks
#                 self.attn2_4 = AttentionBlock(model.layer2[1].conv2.out_channels, model.layer4[1].conv2.out_channels, 128, 4, True)
#                 self.attn3_4 = AttentionBlock(model.layer3[1].conv2.out_channels, model.layer4[1].conv2.out_channels, 128, 2, True)

#             self.last_ftrmap_size = int(self.img_size/(2**5)) #outputsize is the original one divided by 32.
#             self.last_ftrmap_number = 512 #512 for ResNet18

#             if self.causality_aware:

#                 self.causality_map_extractor = CausalityMapBlock(elems=self.last_ftrmap_size**4, causality_method=self.causality_method)

#                 print(f"causality_map_extractor LEAFS:")
#                 print(self.causality_map_extractor.mask.is_leaf)
#                 print()

#                 if self.causality_setting == "cat": #[1, n*n*k + k*k]
#                     self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size + self.last_ftrmap_number*self.last_ftrmap_number, self.num_classes)
#                 elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool"): #[1, 2*n*n*k]
#                     self.relu = nn.ReLU()
#                     self.classifier = nn.Linear(2 * self.last_ftrmap_size * self.last_ftrmap_size * self.last_ftrmap_number, self.num_classes)
#                 elif self.causality_setting == "mul" or self.causality_setting == "mulbool": #TODO 18 settembre
#                     self.relu = nn.ReLU()
#                     self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size, self.num_classes)
#             else:
#                 if self.visual_attention: #TODO hardcoded, considerando attn2_4 e attn3_4, intanto solo per la versione non causale, poi preparare il codice anche per quela causale quindi mettere anche nel IF sopra
#                     self.classifier = nn.Linear(128*12*12 + 256*6*6 + self.last_ftrmap_number*self.last_ftrmap_size*self.last_ftrmap_size, self.num_classes)
#                 else:
#                     self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size, self.num_classes)

#             self.softmax = nn.Softmax(dim=1) #TODO 12 luglio, added

#     def forward(self, x):
#         if torch.isnan(x).any():
#             l_ingresso_era_gia_corrotto_con_nan
#         # print(f"FORWARD - x (input): {x.requires_grad}")

#         # print(f"x[0] (forward):\t MIN:{x[0].min()}\t MAX:{x[0].max()}\t AVG:{torch.mean(x[0].float())}\t MED:{torch.median(x[0].float())}")
#         x = self.starting_block(x)
#         # print(f"x[0] (starting_block):\t MIN:{x[0].min()}\t MAX:{x[0].max()}\t AVG:{torch.mean(x[0].float())}\t MED:{torch.median(x[0].float())}")
#         x_layer1 = self.layer1(x)
#         # print(f"x_layer1  (layer1):\t MIN:{x_layer1[0].min()}\t MAX:{x_layer1[0].max()}\t AVG:{torch.mean(x_layer1[0].float())}\t MED:{torch.median(x_layer1[0].float())}")
#         x_layer2 = self.layer2(x_layer1)
#         # print(f"x_layer2(x_layer2):\t MIN:{x_layer2[0].min()}\t MAX:{x_layer2[0].max()}\t AVG:{torch.mean(x_layer2[0].float())}\t MED:{torch.median(x_layer2[0].float())}")
#         x_layer3 = self.layer3(x_layer2)
#         # print(f"x_layer3(x_layer3):\t MIN:{x_layer3[0].min()}\t MAX:{x_layer3[0].max()}\t AVG:{torch.mean(x_layer3[0].float())}\t MED:{torch.median(x_layer3[0].float())}")
#         x_layer4 = self.layer4(x_layer3)
#         # print(f"x_layer4(x_layer4):\t MIN:{x_layer4[0].min()}\t MAX:{x_layer4[0].max()}\t AVG:{torch.mean(x_layer4[0].float())}\t MED:{torch.median(x_layer4[0].float())}")
#         x = self.ending_block(x_layer4)
#         # print(f"x at -ending_block:\t MIN:{x[0].min()}\t MAX:{x[0].max()}\t AVG:{torch.mean(x[0].float())}\t MED:{torch.median(x[0].float())}")
        
#         # compute the attention probabilities:
#         # a2_4, x2_4 = self.attn2_4(x_layer2, x) #TODO per il momento, le probabilità a_ non le uso, ma serviranno per XAI
#         # a3_4, x3_4 = self.attn3_4(x_layer3, x) usare queste x_ per concatenarle all output da classificare

#         #check nan
#         if torch.isnan(x).any():
#             corrotto_con_nan
#         if list(x.size()) != [int(x.size(0)), self.last_ftrmap_number, self.last_ftrmap_size, self.last_ftrmap_size]:
#             x = torch.reshape(x, (int(x.size(0)), self.last_ftrmap_number, self.last_ftrmap_size, self.last_ftrmap_size))

#         causality_maps = None #initialized to none

#         if self.causality_aware:
#             x, causality_maps = self.causality_map_extractor(x)
#             # print(f"FORWARD - x (causality_map_extractor): {x.requires_grad}")
#             # print(f"FORWARD - causality_maps (causality_map_extractor): {causality_maps.requires_grad}")
            

#             if self.causality_setting == "cat":
#                 x = torch.cat((torch.flatten(x, 1), torch.flatten(causality_maps, 1)), dim=1)
#             # elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool"):
#             elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool") or (self.causality_setting == "mul") or (self.causality_setting == "mulbool"):
#                 b = x.size()[0] # number of images in the batch
#                 k = x.size()[1] # number of feature maps
#                 x_c = torch.zeros((b, k, self.last_ftrmap_size, self.last_ftrmap_size), device=x.get_device())

#                 for n in range(causality_maps.size()[0]): #batch size
#                     causality_map = causality_maps[n]
#                     triu = torch.triu(causality_map, 1) #upper triangular matrx (excluding the principal diagonal)
#                     tril = torch.tril(causality_map, -1).T.contiguous() #lower triangular matrx  (excluding the principal diagonal), transposed for easier elementwise computation with the upper
#                     bool_ij = (tril>triu).T
#                     bool_ji = (triu>tril)
#                     bool_matrix = bool_ij + bool_ji #sum of booleans is the OR logic
#                     by_col = torch.sum(bool_matrix, 1)
#                     by_row = torch.sum(bool_matrix, 0)

#                     if self.MULCAT_CAUSES_OR_EFFECTS=="causes":
#                         # if self.causality_setting == "mulcat":
#                         if self.causality_setting == "mulcat" or self.causality_setting == "mul":
#                             causes_mul_factors = by_col - by_row # the factor of a featuremap is how many times it causes some other featuremap minus how many times it is caused by other feature maps
#                         elif self.causality_setting == "mulcatbool" or self.causality_setting == "mulbool":
#                             causes_mul_factors = 1.0*((by_col - by_row)>0) # the factor of a featuremap is 1 (pass) iff it causes some other featuremap more than how many times itself is caused by other feature maps, 0 (not pass) otherwise
                    
#                     elif self.MULCAT_CAUSES_OR_EFFECTS=="effects": #TODO aggiunto queto if self.MULCAT_CAUSES_OR_EFFECTS: the other way round, take the effects instead
#                         if self.causality_setting == "mulcat" or self.causality_setting == "mul":
#                             causes_mul_factors = by_row - by_col # 
#                         elif self.causality_setting == "mulcatbool" or self.causality_setting == "mulbool":
#                             causes_mul_factors = 1.0*((by_row - by_col)>0) # 
#                     else:
#                         raise ValueError
                
#                     x_causes = torch.einsum('kmn,k->kmn', x[n,:,:,:], causes_mul_factors)#multiply each factor for the corresponding 2D feature map
#                     x_c[n] = self.relu(x_causes) #rectify every negative value to zero

#                 # x = torch.cat((torch.flatten(x, 1), torch.flatten(x_c, 1)), dim=1) #cat the feature maps with their causality-enhanced version
#                 #TODO 18 settembre 2023: commented the above and run the below 
#                 if self.causality_setting == "mul" or self.causality_setting == "mulbool":
#                     x = torch.flatten(x_c, 1) #substitute the actual features with the filtered version of them.
#                 else: #mulcat or mulcatbool need to concatenate the X_c to the actual original features x
#                     x = torch.cat((torch.flatten(x, 1), torch.flatten(x_c, 1)), dim=1) #cat the feature maps with their causality-enhanced version

            
#         else: #traditional, non causal:
#             x = torch.flatten(x, 1) # flatten all dimensions except batch 
        
#         x = self.classifier(x)
#         x = self.softmax(x) #TODO 12 luglio 2023
#         # print(torch.round(x,decimals=2))

#         # print(f"FORWARD - x (return): {x.requires_grad}")
#         # print(f"FORWARD - causality_maps (return): {causality_maps.requires_grad}")
#         return x, causality_maps #return the logit of the classification, and the causality maps for optional visualization or some metric manipulation during training
    


## Convnext models not completely curated yet, utilize the resnet18 version instead. TODO.
# from torchvision.models import convnext_tiny, convnext_base
# from torchvision.models.convnext import ConvNeXt_Tiny_Weights, ConvNeXt_Base_Weights
# class ResNextCA(nn.Module):
#     def __init__(self, which_resnext, dim, channels, num_classes, is_pretrained, is_feature_extractor, causality_aware=False, causality_method="max", LEHMER_PARAM=None, causality_setting="cat", visual_attention=False, MULCAT_CAUSES_OR_EFFECTS="causes"):
            
#             super(ResNextCA, self).__init__()            
                
#             self.which_resnext = which_resnext
#             self.img_size = dim
#             self.channels = channels
#             self.num_classes = num_classes
#             self.is_pretrained = is_pretrained
#             self.is_feature_extractor = is_feature_extractor
            
#             self.causality_aware = causality_aware
#             self.causality_method = causality_method
#             self.LEHMER_PARAM = LEHMER_PARAM
#             self.causality_setting = causality_setting #

#             self.visual_attention = visual_attention #boolean
#             self.MULCAT_CAUSES_OR_EFFECTS = MULCAT_CAUSES_OR_EFFECTS #TODO 21 luglio

#             if self.which_resnext=="tiny":
#                 self.first_output_ftrsmap_number = 96
#                 if self.is_pretrained:
#                     model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
#                 else:
#                     model = convnext_tiny()
#             elif self.which_resnext=="base":
#                 self.first_output_ftrsmap_number = 128
#                 if self.is_pretrained:
#                     model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
#                 else:
#                     model = convnext_base()

#             if self.channels == 1:
#                 model.features[0][0] = nn.Conv2d(1, self.first_output_ftrsmap_number, kernel_size=4, stride=4)
#             elif self.channels==3:
#                 if self.is_feature_extractor:
#                     for param in model.parameters(): #freeze the extraction layers
#                         param.requires_grad = False
            
#             ## creating the structure of our custom model starting from the original building blocks of Resnext:
#             # layer0, layer1, layer2, layer3, layer4, layer5, layer6,layer7, and classifier
#             self.layer0 = model.features[0] #
#             self.layer1 = model.features[1] #
#             self.layer2 = model.features[2] #
#             self.layer3 = model.features[3] #
#             self.layer4 = model.features[4] #
#             self.layer5 = model.features[5] #
#             self.layer6 = model.features[6] #
#             self.layer7 = model.features[7] #

#             model.avgpool = Identity() # Cancel adaptiveavgpool2d layer to get feature maps of size, say, 7x7       
#             self.avgpool = model.avgpool
            
#             ##self.features = model

#             # if self.visual_attention: #TODO commentato per ora, capire quali layer attenzionare a differenza di resnet18
#             #     ## define the attention blocks
#             #     self.attn2_4 = AttentionBlock(model.layer2[1].conv2.out_channels, model.layer4[1].conv2.out_channels, 128, 4, True)
#             #     self.attn3_4 = AttentionBlock(model.layer3[1].conv2.out_channels, model.layer4[1].conv2.out_channels, 128, 2, True)

#             self.last_ftrmap_size = int(self.img_size/(2**5)) #outputsize is the original one divided by 32.
#             if self.which_resnext=="tiny":
#                 self.last_ftrmap_number = 768
#             elif self.which_resnext=="base":
#                 self.last_ftrmap_number = 1024

#             if self.causality_aware:
#                 if self.causality_setting == "cat": #[1, n*n*k + k*k]
#                     self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size + self.last_ftrmap_number*self.last_ftrmap_number, self.num_classes)
#                 elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool"): #[1, 2*n*n*k]
#                     self.relu = nn.ReLU()
#                     self.classifier = nn.Linear(2 * self.last_ftrmap_size * self.last_ftrmap_size * self.last_ftrmap_number, self.num_classes)
#                 elif (self.causality_setting == "mul") or (self.causality_setting == "mulbool"): #TODO 
#                     self.relu = nn.ReLU()
#                     self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size, self.num_classes)
#             else: 
#                 if self.visual_attention: ##TODO False per ora, capire quali dimensioni al posto di 12 e 6 in base ai layer che si scelgono; TODO hardcoded, considerando attn2_4 e attn3_4, intanto solo per la versione non causale, poi preparare il codice anche per quela causale quindi mettere anche nel IF sopra
#                     self.classifier = nn.Linear(128*12*12 + 256*6*6 + self.last_ftrmap_number*self.last_ftrmap_size*self.last_ftrmap_size, self.num_classes)
#                 else:
#                     self.classifier = nn.Linear(self.last_ftrmap_number * self.last_ftrmap_size * self.last_ftrmap_size, self.num_classes)

#             self.softmax = nn.Softmax(dim=1) #TODO 12 luglio, added

#     def forward(self, x):
        
#         if torch.isnan(x).any():
#             l_ingresso_era_gia_corrotto_con_nan

#         # print(f"resnext feature size:\t {x.size()}") #torch.Size([N, 1, 96, 96]) #esempio
#         x_layer0 = self.layer0(x)
#         # print(f"resnext x_layer0 size:\t {x_layer0.size()}") torch.Size([N, 128, 24, 24]) ##esempio, valori con ResNext BASE
#         x_layer1 = self.layer1(x_layer0)
#         # print(f"resnext x_layer1 size:\t {x_layer1.size()}") torch.Size([N, 128, 24, 24])
#         x_layer2 = self.layer2(x_layer1)
#         # print(f"resnext x_layer2 size:\t {x_layer2.size()}")  torch.Size([N, 256, 12, 12])
#         x_layer3 = self.layer3(x_layer2)
#         # print(f"resnext x_layer3 size:\t {x_layer3.size()}")  torch.Size([N, 256, 12, 12])
#         x_layer4 = self.layer4(x_layer3)
#         # print(f"resnext x_layer4 size:\t {x_layer4.size()}") torch.Size([N, 512, 6, 6])
#         x_layer5 = self.layer5(x_layer4)
#         # print(f"resnext x_layer5 size:\t {x_layer5.size()}") torch.Size([N, 512, 6, 6])
#         x_layer6 = self.layer6(x_layer5)
#         # print(f"resnext x_layer6 size:\t {x_layer6.size()}") torch.Size([N, 1024, 3, 3])
#         x_layer7 = self.layer7(x_layer6)
#         # print(f"resnext x_layer7 size:\t {x_layer7.size()}") torch.Size([N, 1024, 3, 3])
#         x = self.avgpool(x_layer7)
#         # print(f"resnext avgpool size:\t {x.size()}") #torch.Size([N, 1024, 3, 3]) <-- siccome avevo messo Identity
        

#         # compute the attention probabilities:
#         # a2_4, x2_4 = self.attn2_4(x_layer2, x) #TODO per il momento, le probabilità a_ non le uso, ma serviranno per XAI
#         # a3_4, x3_4 = self.attn3_4(x_layer3, x) usare queste x_ per concatenarle all output da classificare

#         #check nan
#         if torch.isnan(x).any():
#             corrotto_con_nan
#         if list(x.size()) != [int(x.size(0)), self.last_ftrmap_number, self.last_ftrmap_size, self.last_ftrmap_size]:
#             print("Correcting shape mismatch...(in forward, after avgpool, before causality module)")
#             x = torch.reshape(x, (int(x.size(0)), self.last_ftrmap_number, self.last_ftrmap_size, self.last_ftrmap_size))

#         causality_maps = None #initialized to none

#         if self.causality_aware:
#             # x, causality_maps = self.get_causality_maps(x) # code for computing causality maps given a batch of featuremaps x
#             x, causality_maps = GetCausalityMaps(x, self.causality_method, self.LEHMER_PARAM)

#             if self.causality_setting == "cat":
#                 x = torch.cat((torch.flatten(x, 1), torch.flatten(causality_maps, 1)), dim=1)
#             elif (self.causality_setting == "mulcat") or (self.causality_setting == "mulcatbool") or (self.causality_setting == "mul") or (self.causality_setting == "mulbool"):
#                 b = x.size()[0] # number of images in the batch
#                 k = x.size()[1] # number of feature maps
#                 x_c = torch.zeros((b, k, self.last_ftrmap_size, self.last_ftrmap_size), device=x.get_device())

#                 for n in range(causality_maps.size()[0]): #batch size
#                     causality_map = causality_maps[n]
#                     triu = torch.triu(causality_map, 1) #upper triangular matrx (excluding the principal diagonal)
#                     tril = torch.tril(causality_map, -1).T.contiguous() #lower triangular matrx  (excluding the principal diagonal), transposed for easier elementwise computation with the upper
#                     bool_ij = (tril>triu).T.contiguous()
#                     bool_ji = (triu>tril)
#                     bool_matrix = bool_ij + bool_ji #sum of booleans is the OR logic
#                     by_col = torch.sum(bool_matrix, 1)
#                     by_row = torch.sum(bool_matrix, 0)

#                     if self.MULCAT_CAUSES_OR_EFFECTS=="causes":
#                         if self.causality_setting == "mulcat" or self.causality_setting == "mul":
#                             causes_mul_factors = by_col - by_row # the factor of a featuremap is how many times it causes some other featuremap minus how many times it is caused by other feature maps
#                         elif self.causality_setting == "mulcatbool" or self.causality_setting == "mulbool":
#                             causes_mul_factors = 1.0*((by_col - by_row)>0) # the factor of a featuremap is 1 (pass) iff it causes some other featuremap more than how many times itself is caused by other feature maps, 0 (not pass) otherwise
                    
#                     elif self.MULCAT_CAUSES_OR_EFFECTS=="effects": #TODO aggiunto queto if self.MULCAT_CAUSES_OR_EFFECTS: the other way round, take the effects instead
#                         if self.causality_setting == "mulcat" or self.causality_setting == "mul":
#                             causes_mul_factors = by_row - by_col # 
#                         elif self.causality_setting == "mulcatbool" or self.causality_setting == "mulbool":
#                             causes_mul_factors = 1.0*((by_row - by_col)>0) # 
#                     else:
#                         raise ValueError
                
#                     x_causes = torch.einsum('kmn,k->kmn', x[n,:,:,:], causes_mul_factors)#multiply each factor for the corresponding 2D feature map
#                     x_c[n] = self.relu(x_causes) #rectify every negative value to zero
                    
#                 # x = torch.cat((torch.flatten(x, 1), torch.flatten(x_c, 1)), dim=1) #cat the feature maps with their causality-enhanced version
#                 #TODO commented the above and run the below 
#                 if self.causality_setting == "mul" or self.causality_setting == "mulbool":
#                     x = torch.flatten(x_c, 1) #substitute the actual features with the filtered version of them.
#                 else: #mulcat or mulcatbool need to concatenate the X_c to the actual original features x
#                     x = torch.cat((torch.flatten(x, 1), torch.flatten(x_c, 1)), dim=1) #cat the feature maps with their causality-enhanced version

#         else: #traditional, non causal:
#             x = torch.flatten(x, 1) # flatten all dimensions except batch 
        
#         x = self.classifier(x)
#         x = self.softmax(x) #TODO 12 luglio 2023
#         # print(x)
#         return x, causality_maps #return the logit of the classification, and the causality maps for optional visualization or some metric manipulation during training
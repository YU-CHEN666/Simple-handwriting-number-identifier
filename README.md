# 目錄
- [簡介](#簡介)
- [操作方法](#操作方法)
   - [關閉警告、結果視窗](#關閉警告、結果視窗)
   - [會觸發警告的範例](#會觸發警告的範例)
   - [關於數字7的注意事項](#關於數字7的注意事項)
   - [辨識操作程式不會有反應的情況](#辨識操作程式不會有反應的情況)
- [模型](#模型)
   - [模型架構比較](#模型架構比較)
   - [架構1訓練過程](#架構1訓練過程)
   - [架構2訓練過程](#架構2訓練過程)
   - [架構3訓練過程](#架構3訓練過程)
- [使用到的模組及版本](#使用到的模組及版本)
# 簡介
 使用OpenCV來獲取手寫數字，搭配預先訓練好的神經網路模型，進行辨識再顯示結果，一個簡易的手寫數字辨識器就此誕生 :exclamation: :exclamation: :exclamation: :exclamation: :exclamation:
# 操作方法
 執行程式後，會有操作說明視窗，請遵循說明操作。
> [!IMPORTANT]
> 在執行鍵盤操作前，請先確定輸入法是**英文模式(打出英文字母)**，這樣程式才會有反應。
<a name="關閉警告、結果視窗"></a>
<details>
 <summary>關閉警告、結果視窗(點擊展開內容)</summary>

 有以下兩種方式:
 1. 按下視窗右上角的X。![image](/picture/x按鈕.png)
 2. 按下任意鍵
 **(推薦使用此方式)**。\
 **若使用方式1的話，_所有鍵盤操作必須按2次，程式才會有反應_**，因此建議使用方法2。

</details>
<a name="會觸發警告的範例"></a> 
<details>                           
 <summary>會觸發警告的範例</summary>

 1. 一位數時:書寫數字5的過程不連續，有斷點。\
 ![image](/picture/觸發警告範例/1-0.bmp)
 2. 二位數時:數字歪斜+間隔過近。\
 ![image](/picture/觸發警告範例/2-0.png)

</details>
<a name="關於數字7的注意事項"></a>
<details> 
   <summary>關於數字7的注意事項</summary>

   1. 一位數時:書寫過程不連續有斷點，**絕對會辨識錯誤**，辨識成二位數。\
   ![image](/picture/數字7注意事項/1.bmp)
   2. 二位數時:此情況下進行辨識操作，程式不會有任反應，直到你清除重寫，再進行辨識操作。\
   ![image](/picture/數字7注意事項/2.png)
    
</details>
<a name="辨識操作程式不會有反應的情況"></a>
<details>
 <summary>辨識操作程式不會有反應的情況</summary>

 1. 輸入法為中文模式，程式無法偵測到你給予的指令。
 2. 程式判定目前書寫的數字超過2位數，幾使是一點程式仍然會判定為一位。\
 ![image](/picture/沒反應情況/0.png) ![image](/picture/沒反應情況/1.png)

</details>

# 模型
<a name="模型架構比較"></a> 
<details>
   <summary>模型架構比較</summary>

   ![image](/picture/架構比較.bmp)\
   :large_orange_diamond: 架構設計順序: 架構1:arrow_right:架構2:arrow_right:架構3\
   :large_orange_diamond: 對於ELAN:
   - 架構1: 使用1x1捲積進行分割。
   - 架構2&3: 使用自定義層進行分割，而且不涉及任何參數的訓練。

</details>
<a name="架構1訓練過程"></a>
<details>
 <summary>架構1訓練過程</summary>
   
 1. 使用Adam+學習率計畫(餘弦重啟)，因為只設定20個Epoch，只能先停止，準確率變化如下圖。\
 ![image](/picture/架構1/first.jpg)
 2. 接續訓練，嘗試使用SGD(不同的學習率、不同的動量值、權重衰減不同強度)+有無學習率計畫(餘弦重啟、Epoch衰減)、上一段的設定，最終以上一段的設定表現最佳。準確率變化如下圖。\
 ![image](/picture/架構1/接續.bmp)

</details>
<a name="架構2訓練過程"></a>
<details>
   <summary>架構2訓練過程</summary>

   1. 使用AdamW(weight_decay=0.0005)+學習率計畫(餘弦重啟)，準確率變化如下圖。\
   ![image](/picture/架構2/first.bmp)
   2. 取上一段第13個Epoch的模型接續訓練，使用SGD(相同學習率、相同動量值、權重衰減不同強度)+有無學習率計畫(每個Epoch衰減0.5)。參數表現最佳的準確率變化如下圖。\
   ![image](/picture/架構2/接續.bmp)

</details>
<a name="架構3訓練過程"></a>
<details>
  <summary>架構3訓練過程</summary>

 1. 使用AdamW(weight_decay=0.003)+學習率計畫(每個Epoch衰減0.6)。準確率變化如下圖。\
 ![image](/picture/架構3/first.bmp)
 2. 取上一段第7個Epoch的模型接續訓練，使用SGD(learning_rate=0.05,momentum=0.4,weight_decay=0.005)+學習率計畫(每個Epoch衰減0.6)。準確率變化如下圖。\
 ![image](/picture/架構3/接續.bmp)

</details>

:white_check_mark:最終決定選擇，架構2紅色框框的Epoch做為辨識器的模型。
# 使用到的模組及版本
:atom: Python=3.12(我大膽猜測更低的版本也行，因為主要都是靠模組在操作)
|模組|版本|
|:---:|:---:|
|numpy|1.26.4|
|opencv|4.10.0.84|
|tensorflow|2.17.0|
|keras|3.5.0 ~ 3.9.2|



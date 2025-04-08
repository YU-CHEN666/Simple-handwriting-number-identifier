# 目錄
- [簡介](#簡介)
- [操作方法](#操作方法)
   - [關閉警告、結果視窗](#關閉警告、結果視窗)
   - [會觸發警告的範例](#會觸發警告的範例)
   - [關於數字7的注意事項](#關於數字7的注意事項)
   - [辨識操作程式不會有反應的情況](#辨識操作程式不會有反應的情況)
- [模型](#模型)
   - [模型架構比較](#模型架構比較)
- [使用到的模組及版本](#使用到的模組及版本)
# 簡介
 使用OpenCV來獲取手寫數字，搭配預先訓練好的神經網路模型，進行辨識再顯示結果，一個簡易的手寫數字辨識器就此誕生 :exclamation: :exclamation: :exclamation:
# 操作方法
 執行程式後，會有操作說明視窗，請遵循說明操作。
> [!IMPORTANT]
> 在執行鍵盤操作前，請先確定輸入法是**英文模式(打出英文字母)**，這樣程式才會有反應。
<details>
 <a name="關閉警告、結果視窗"></a>
 <summary>關閉警告、結果視窗(按下最左邊的三角型展開內容)</summary>

 有以下兩種方式:\
 1. 按下視窗右上角的X。![image](/picture/x按鈕.png)
 2. 按下任意鍵
 **(推薦使用此方式)**。\
 **若使用方式1的話，_所有鍵盤操作必須按2次，程式才會有反應_**，因此建議使用方法2。

</details>
<details>
 <a name="會觸發警告的範例"></a>                            
 <summary>會觸發警告的範例</summary>

 1. 一位數時:書寫數字5的過程不連續，有斷點。\
 ![image](/picture/觸發警告範例/1-0.bmp)
 2. 二位數時:數字歪斜+間隔過近。\
 ![image](/picture/觸發警告範例/2-0.png)

</details>
<details>
   <a name="關於數字7的注意事項"></a> 
   <summary>關於數字7的注意事項</summary>

   1. 一位數時:書寫過程不連續有斷點，**絕對會辨識錯誤**，辨識成二位數。\
   ![image](/picture/數字7注意事項/1.bmp)
   2. 二位數時:此情況下進行辨識操作，程式不會有任反應，直到你清除重寫，再進行辨識操作。\
   ![image](/picture/數字7注意事項/2.png)
    
</details>
<details>
 <a name="辨識操作程式不會有反應的情況"></a>
 <summary>辨識操作程式不會有反應的情況</summary>

 1. 輸入法為中文模式，程式無法偵測到你給予的指令。
 2. 程式判定目前書寫的數字超過2位數，幾使是一點程式仍然會判定為一位。\
 ![image](/picture/沒反應情況/0.png) ![image](/picture/沒反應情況/1.png)

</details>

# 模型
<details>
   <a name="模型架構比較"></a> 
   <summary>模型架構比較</summary>

   ![image](/picture/架構比較.bmp)\
   :large_orange_diamond: 架構設計順序: 架構1:arrow_right:架構2:arrow_right:架構3\
   :large_orange_diamond:1321
   
</details>

# 使用到的模組及版本

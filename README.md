# 揭開黑箱模型：探索可解釋人工智慧

> [第15屆iT邦幫忙鐵人賽](https://ithelp.ithome.com.tw/users/20107247/ironman/6272)

本系列將從 XAI 的基礎知識出發，深入探討可解釋人工智慧在機器學習和深度學習中的應用、案例和挑戰，以及未來發展方向。希望透過這個系列，幫助讀者更好地理解和應用可解釋人工智慧技術，促進可信、透明、負責任的人工智慧發展。

#### 鐵人賽列表

| 文章 | 程式 |
| ------------- | ------------- |
| [[Day 1] 揭開模型的神秘面紗：為何XAI對機器學習如此重要？](https://ithelp.ithome.com.tw/articles/10318087) | -  |
| [[Day 2] 從黑盒到透明化：XAI技術的發展之路](https://ithelp.ithome.com.tw/articles/10318532) | -  |
| [[Day 3] 機器學習中的可解釋性指標](https://ithelp.ithome.com.tw/articles/10319364) | -  |
| [[Day 4] LIME vs. SHAP：哪種XAI解釋方法更適合你？](https://ithelp.ithome.com.tw/articles/10320360) | -  |
| [[Day 5] 淺談XAI與傳統機器學習的區別](https://ithelp.ithome.com.tw/articles/10321697) | -  |
| [[Day 6] 非監督學習也能做到可解釋性？探索XAI在非監督學習中的應用](https://ithelp.ithome.com.tw/articles/10322594) | [Code](https://colab.research.google.com/github/andy6804tw/2023-15th-ironman/blob/main/06.非監督學習也能做到可解釋性？探索XAI在非監督學習中的應用.ipynb)  |
| [[Day 7] KNN與XAI：從鄰居中找出模型的決策邏輯](https://ithelp.ithome.com.tw/articles/10323663) | [Code](https://colab.research.google.com/github/andy6804tw/2023-15th-ironman/blob/main/07.KNN與XAI：從鄰居中找出模型的決策邏輯.ipynb)  |
| [[Day 8] 解釋線性模型：探索線性迴歸和邏輯迴歸的可解釋性](https://ithelp.ithome.com.tw/articles/10324299) | [Code](https://colab.research.google.com/github/andy6804tw/2023-15th-ironman/blob/main/08.解釋線性模型：探索線性迴歸和邏輯迴歸的可解釋性.ipynb)  |
| [[Day 9] 基於樹狀結構的XAI方法：決策樹的可解釋性](https://ithelp.ithome.com.tw/articles/10325159) | [Code](https://colab.research.google.com/github/andy6804tw/2023-15th-ironman/blob/main/09.基於樹狀結構的XAI方法：決策樹的可解釋性.ipynb)  |
| [[Day 10] Permutation Importance：從特徵重要性角度解釋整個模型行為](https://ithelp.ithome.com.tw/articles/10325613) | [Code](https://colab.research.google.com/github/andy6804tw/2023-15th-ironman/blob/main/10.Permutation%20Importance：從特徵重要性角度解釋整個模型行為.ipynb)  |
| [[Day 11] Partial Dependence Plot：探索特徵對預測值的影響](https://ithelp.ithome.com.tw/articles/10326424) | [Code](https://colab.research.google.com/github/andy6804tw/2023-15th-ironman/blob/main/11.Partial%20Dependence%20Plot：探索特徵對預測值的影響.ipynb)  |
| [[Day 12] LIME理論：如何用局部線性近似解釋黑箱模型](https://ithelp.ithome.com.tw/articles/10327698) | -  |
| [[Day 13] LIME實作：實戰演練LIME解釋方法](https://ithelp.ithome.com.tw/articles/10328780) | [Code](https://colab.research.google.com/github/andy6804tw/2023-15th-ironman/blob/main/13.LIME實作：實戰演練LIME解釋方法.ipynb)  |
| [[Day 14] SHAP理論：解析SHAP解釋方法的核心](https://ithelp.ithome.com.tw/articles/10329606) | -  |
| [[Day 15] SHAP實作：實戰演練SHAP解釋方法](https://ithelp.ithome.com.tw/articles/10330115) | [Code](https://colab.research.google.com/github/andy6804tw/2023-15th-ironman/blob/main/15.SHAP實作：實戰演練SHAP解釋方法.ipynb)   |
| [[Day 16] 神經網路的可解釋性：如何理解深度學習中的黑箱模型？](https://ithelp.ithome.com.tw/articles/10330576) | -  |
| [[Day 17] 解析深度神經網路：使用Deep SHAP進行模型解釋](https://ithelp.ithome.com.tw/articles/10331443) | [Code](https://colab.research.google.com/github/andy6804tw/2023-15th-ironman/blob/main/17.解析深度神經網路：使用Deep%20SHAP進行模型解釋.ipynb)  |
| [[Day 18] CNN：卷積深度神經網路的解釋方法](https://ithelp.ithome.com.tw/articles/10332039) | -  |
| [[Day 19] Perturbation-Based：如何用擾動方法解釋神經網路](https://ithelp.ithome.com.tw/articles/10332904) | [Code](https://colab.research.google.com/github/andy6804tw/2023-15th-ironman/blob/main/19.Perturbation-Based：如何用擾動方法解釋神經網路.ipynb)  |
| [[Day 20] Gradient-Based：利用梯度訊息解釋神經網路](https://ithelp.ithome.com.tw/articles/10333578) | [Code](https://colab.research.google.com/github/andy6804tw/2023-15th-ironman/blob/main/20.Gradient-Based：利用梯度訊息解釋神經網路.ipynb)  |
| [[Day 21] Propagation-Based：探索反向傳播法的可解釋性](https://ithelp.ithome.com.tw/articles/10334191) | -  |
| [[Day 22] CAM-Based：如何解釋卷積神經網路](https://ithelp.ithome.com.tw/articles/10334625) | [Code](https://colab.research.google.com/github/andy6804tw/2023-15th-ironman/blob/main/22.CAM-Based：如何解釋卷積神經網路.ipynb)  |
| [[Day 23] Attention-Based：使用注意力機制解釋CNN模型](https://ithelp.ithome.com.tw/articles/10335422) | []()  |
| [[Day 24] LSTM的可解釋性：從時序資料解析人體姿態預測](https://ithelp.ithome.com.tw/articles/10335915) | [Code](https://colab.research.google.com/github/andy6804tw/2023-15th-ironman/blob/main/24.LSTM的可解釋性：從時序資料解析人體姿態預測.ipynb)  |
| [[Day 25] XAI在影像處理中的瑕疵檢測：解釋卷積神經網路的運作]() | [Code](https://www.kaggle.com/code/andy6804tw/day-25-xai)  |
| [[Day 26] XAI在表格型資料的應用：解析智慧工廠中的鋼材缺陷](https://ithelp.ithome.com.tw/articles/10337150) | [Code](https://colab.research.google.com/github/andy6804tw/2023-15th-ironman/blob/main/26.XAI在表格型資料的應用：解析智慧工廠中的鋼材缺陷.ipynb)  |
| [[Day 27] XAI在NLP中的應用：以情感分析解釋語言模型](https://ithelp.ithome.com.tw/articles/10337606) | [Code](https://colab.research.google.com/github/andy6804tw/2023-15th-ironman/blob/main/27.XAI在NLP中的應用：以情感分析解釋語言模型.ipynb)  |
| [[Day 28] XAI如何影響人類對技術的信任和接受程度？](https://ithelp.ithome.com.tw/articles/10338219) | -  |
| [[Day 29] 對抗樣本的挑戰：如何利用XAI檢測模型的弱點？](https://ithelp.ithome.com.tw/articles/10338669) | -  |
| [[Day30] XAI未來發展方向：向更可靠的機器學習模型邁進](https://ithelp.ithome.com.tw/articles/10339196) | -  |
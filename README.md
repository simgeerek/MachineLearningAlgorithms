# Makine Öğrenmesi Algoritmaları

## Basit Doğrusal Regresyon

Basit doğrusal regresyon, bir bağımlı değişken ve bir bağımsız değişkenin arasındaki doğrusal ilişkinin modellenmesidir.
Şu şekilde ifade edilir:

![alt text](https://cdn-images-1.medium.com/max/1600/1*5j7BZRiy7GRJRbZImCkpEg.png)

x değişkeni bağımsız değişken, y değişkeni x değişkenine bağlı olarak değişen bağımlı değişkendir. β0 ve β1 değerleri modelin bilinmeyen katsayılarıdır. β0 regresyon doğrusunun y eksenini kestiği noktayı, β1 ise x bağımsız değişkeninde meydana gelen birim değişikliğin y bağımlı değişkenini etkileyeceği miktarı verir. ε, hata terimi(error term), gözlemlenen verilerin gerçek verilerden nasıl farklı olduğunu temsil eden değerdir.

Amaç;
- Bağımlı y değişkeni ile bağımsız x değişkeni arasında ilişkiyi tanımlayan basit doğrusal denklemi elde etmek,
- x'de birim artış olduğunda y'deki değişiklik miktarı olan katsayıyı bulmak,
- Elde edilen matematiksek denklem yardımıyla x değişkenin değeri ile bilinmeyen y değişkenin değerini kestirmektir.

Peki x ve y değişkeni arasındaki ilişkiyi en iyi tanımlayan katsayı değerlerini nasıl bulacağız? En küçük kareler yöntemi, değişkenler arasındaki ilişkiyi tanımlayan en uyumlu doğrusal denklemi bulmak, gerçek değerler ile tahmin edilen değerler arasındaki hata terimini minimuma indirmek için kullanılır. 
En küçük kareler yöntemi ile bulunan tahmini regresyon denkleminin ifade edilişi :

![alt text](https://cdn-images-1.medium.com/max/1600/1*KnyqulfXF3_LdddjjCwFCQ.png)

ŷ belirli bir x bağımsız değişkeni için tahmin edilen y bağımlı değişkeni, β0 ve β1 en küçük kareler yöntemiyle bulunan hatayı minimuma indiren katsayılardır.


## Çoklu Doğrusal Regresyon

Çoklu doğrusal regresyon, bir bağımlı değişken ve birden fazla bağımsız değişkenin arasındaki doğrusal ilişkinin modellenmesidir.
![alt text](https://cdn-images-1.medium.com/max/1600/1*wfPYBWtYAXcp8k4xCKBYEQ.png)

x1, x2,.., xn değişkenleri bağımsız değişkenler, y değişkeni x değişkenlerine bağlı olarak değişen bağımlı değişkendir. β0 kesim noktasıdır, tüm x değişkenleri sıfır olduğunda bağımlı değişkenin aldığı değerdir. β1, β2,.., βn katsayıları bir x değişkeni için diğer x değişkenleri sabit tutulduğunda beklenen değişiklik miktarı yani y değişkenini etkileyen ağırlıktır. ε hata terimidir.

Amaç;
- Bağımlı y değişkeni ile bağımsız x değişkenleri arasında ilişkiyi tanımlayan karmaşık doğrusal denklemi elde etmek,
- Tüm değişkenlerin varlığında katsayıları bulmak,
- Elde edilen matematiksek denklem yardımıyla x değişkenlerinin değerleri ile bilinmeyen y değişkenin değerini kestirmek,
- Bağımlı değişkeni etkileyen bağımsız değişkenlerden hangilerinin bağımlı değişkeni daha çok etkilediğini bulmak.

En küçük kareler yöntemi ile katsayıların hesaplanmasıyla bulunan tahmini çoklu regresyon denkleminin ifade edilişi :
![alt text](https://cdn-images-1.medium.com/max/1600/1*ZtCrkEBUkx_LZOuggh-EqQ.png)

ŷ belirli x değişkenleri için tahmin edilen y bağımlı değişkenidir.

## Regresyon Hata Metrikleri

Basit doğrusal regresyon ve çoklu doğrusal regresyon modelleri kurup, bağımlı değişkenleri tahminledik. Peki bu kurduğumuz modellerin başarısını nasıl değerlendireceğiz? Regresyon modellerinin tahmin başarısını değerlendirmek için bazı metriklerler kullanmakta; R Square, MSE(Mean Square Error), RMSE(Root Mean Square Error), MEA(Mean Absolute Error).

**MSE**, tahmin edilen ve gerçek değer arasındaki farkın karesinin ortalamasıdır. Türevlenebilir ve dışbükey bir şekle sahip olduğu için optimize etmesi daha kolaydır.
![alt text](https://cdn-images-1.medium.com/max/1600/1*5Wi49XWG7kswMTGr7uBPEg.png)
**MEA**, gerçek değer iletahmin edilen değer arasındaki mutlak farkın ortalamasıdır. Aykırı değerlerin öne çıktığı durumlarda tercih edilmez.
![alt text](https://cdn-images-1.medium.com/max/1600/1*_dG1PmMW8up66VDHDexEXw.png)
**RMSE**, tahmin edilen ve gerçek değer arasındaki farkın karesinin ortalamasının kareköküdür.
![alt text](https://cdn-images-1.medium.com/max/1600/1*suGwo5i5J8l4sutHLJgw-Q.png)
**R Square**, bağımsız değişkenlerin bağımlı değişkendeki varyans yüzdesini gösterir. Model ile bağımlı değişken arasındaki ilişkinin gücünü % 0–100 uygun bir ölçekte ölçer.

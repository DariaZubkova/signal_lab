## signal_lab
Отчет лежит в Report_signal.pdf
# Постановка задачи

На вход подается изображение, где объектом A является большая коробка, а объектом B - маленькая коробка.

Требуется определить сможет ли маленькая коробочка поместиться в большую, если перемещать маленькую коробочку только параллельным переносом.

# Требования к задаче

* На изображениях имеет одна большая коробка и еще одна коробка меньшего размера.
* Фотографии сделаны на камеру не меньше 8МП, на расстоянии не больше чем 50см.
* Оба объекта лежат на одной поверхности.
* Объекты хорошо освещены, без засветов.
* На изображении должно быть видно отверстие большой коробки.
* Оба объекта полностью находятся в кадре.
* Объекты не касаются друг друга.
* На изображении на заднем фоне не должно быть лишних объектов.
* На большую коробку поставлены 2 розовые метки.

# План

* Для распознавания объекта B используется метод Canny, сегментация, сглаживание, морфология и контур.
* По контору осуществляется поиск максимумов и минимумов по x и y. 
* Из максимумов и минимумом по х и у вычисляется длина и ширина коробки.
* Для распознавания и определения размеров объекта A используется поиск по розовому цвету - ищем метки. Находим их контур.
* По контуру меток осуществляется поиск их максимумов и минимумов по x и y. 
* Из максимумов и минимумом по х и у вычисляется длина и ширина коробки.
* Сравниваются результаты. Соответственно, если ширина и длина объекта A больше, чем размеры объекта B, то он помещается внутрь.

# Датасет

Датасет был разделен на две части: Да - объект В помещается в объект А, и Нет - объект В не помещается в объект А. Ссылка на датасет: https://drive.google.com/drive/folders/1JZkRlRwLvT2SWQ4msF24DE-JdGw6eoWa?usp=sharing

![example](https://user-images.githubusercontent.com/56001699/113474989-c90ea300-947b-11eb-88aa-b21c4a12588c.png)

# Результаты

* Алгоритм выдает правильный результат не для всех входных данных - для нескольких изборажений неверно распознаются нужные точки, что приводит на выходе к ошибке.
* По постановке задачи объекты на изображении должны быть хорошо освещены. На некоторых данных это условие не соблюдается, хоть программа и выдает правильный ответ: метка находятся, но из-за тени "съедается", что уменьшает точность вычислений.
* Во всех остальных случаях работа проходит корректно.

Точность алгоритма: 0.9230769230769231

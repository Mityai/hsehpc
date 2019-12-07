## Лабораторная работа 4. CUDA. Захаров Дмитрий, 165 группа

### Структура:
```
- blur – исходный код свертки и размытия
- converter – утилита для конвертации картинки в массивы RGB-каналов и наобороn
- imgs_srcs – исходные картинки
- results – результаты конвертации картинок в rgb каналы и результаты размытия
plots.ipynb – графики сравнения
```

### Результаты запусков:

#### Картинка: img_srcs/putin_large.jpg

##### Сверточная матрица 3x3
```
height: 1000
width: 1000
blur kernel size: 3x3
Time (CPU): 0.069 s
Time (GPU): 0.163 s
```
---
```
height: 2000
width: 2000
blur kernel size: 3x3
Time (CPU): 0.314 s
Time (GPU): 0.183 s
```
---

```
height: 2500
width: 2500
blur kernel size: 3x3
Time (CPU): 0.49 s
Time (GPU): 0.226 s
```
---
```
height: 2671
width: 4006
blur kernel size: 3x3
Time (CPU): 0.84 s
Time (GPU): 0.276 s
```
---

##### Сверточная матрица 9x9

```
height: 1000
width: 1000
blur kernel size: 9x9
Time (CPU): 0.602 s
Time (GPU): 0.173 s
```

---

```
height: 2000
width: 2000
blur kernel size: 9x9
Time (CPU): 1.954 s
Time (GPU): 0.195 s
```

---

```
height: 2500
width: 2500
blur kernel size: 9x9
Time (CPU): 3.777 s
Time (GPU): 0.245 s
```

---

```
height: 2671
width: 4006
blur kernel size: 9x9
Time (CPU): 6.448 s
Time (GPU): 0.416 s
```

#### Сверточная матрица 15x15
```
height: 1000
width: 1000
blur kernel size: 15x15
Time (CPU): 1.651 s
Time (GPU): 0.237 s
```
---
```
height: 2000
width: 2000
blur kernel size: 15x15
Time (CPU): 6.642 s
Time (GPU): 0.309 s
```
---
```
height: 2500
width: 2500
blur kernel size: 15x15
Time (CPU): 10.351 s
Time (GPU): 0.306 s
```
---
```
height: 2671
width: 4006
blur kernel size: 15x15
Time (CPU): 16.755 s
Time (GPU): 0.351 s
```
---

#### Сверточная матрица 21x21
```
height: 1000
width: 1000
blur kernel size: 21x21
Time (CPU): 3.186 s
Time (GPU): 0.252 s
```
---
```
height: 2000
width: 2000
blur kernel size: 21x21
Time (CPU): 12.729 s
Time (GPU): 0.354 s
```
---
```
height: 2500
width: 2500
blur kernel size: 21x21
Time (CPU): 19.849 s
Time (GPU): 0.578 s
```
---
```
height: 2671
width: 4006
blur kernel size: 21x21
Time (CPU): 34.132 s
Time (GPU): 0.541 s
```
---

#### Графики
https://github.com/Mityai/hsehpc/blob/master/plots.ipynb

### Пример размытой картинки
#### Картинка: img_srcs/putin.jpg
Исходная:
![original](https://github.com/Mityai/hsehpc/blob/master/img_srcs/putin.jpg "Политик, лидер и боец!")

Размытая results/puting_blured.jpg:
![blured](https://github.com/Mityai/hsehpc/blob/master/results/putin_blured.jpg "Политик, лидер и боец!")

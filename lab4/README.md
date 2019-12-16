# Лабораторная работа 4. MPI. Захаров Дмитрий, 165 группа

## Структура:
```
- actual - исходный код, считающий уравнение по точной формуле (количество шагов в суммиировании ограничено)
- solution - параллельное решение с использованием MPI
- plots - графики
```

## Корректность работы

Запуск параллельного решения:
```
$ mpirun -np 4 ./sol 1 0.0002 0.02 500
k = 1, tau = 0.0002, h = 0.02, steps = 500, points = 51
k * tau / h^2 = 0.5
x = 0.00, value = 0.00000000
x = 0.10, value = 0.14668951
x = 0.20, value = 0.27843565
x = 0.30, value = 0.38393650
x = 0.40, value = 0.45040099
x = 0.50, value = 0.47449396
x = 0.60, value = 0.45040099
x = 0.70, value = 0.38393650
x = 0.80, value = 0.27843565
x = 0.90, value = 0.14668951
x = 1.00, value = 0.00000000
```

Запуск точного решения (количество итераций суммирования ограничено 10000 шагами):
```
$ ./sol 1 1 0.1 0.1
x = 0.00, value = 0.00000000
x = 0.10, value = 0.14669054
x = 0.20, value = 0.27898737
x = 0.30, value = 0.38393427
x = 0.40, value = 0.45128579
x = 0.50, value = 0.47448746
x = 0.60, value = 0.45128579
x = 0.70, value = 0.38393427
x = 0.80, value = 0.27898737
x = 0.90, value = 0.14669054
x = 1.00, value = 0.00000000
```

Ошибка < 0.01

## Сравнение времени работы

Графики: https://github.com/Mityai/hsehpc/blob/master/lab4/plots/plots.ipynb

### 2000 точек

<details><summary>1 процесс</summary>

```
$ time mpirun -np 1 ./sol 0.0001 0.0002 0.0005 500000
k = 0.0001, tau = 0.0002, h = 0.0005, steps = 500000, points = 2001
k * tau / h^2 = 0.08
x = 0.00, value = 0.00000000
x = 0.10, value = 0.52049972
x = 0.20, value = 0.84270046
x = 0.30, value = 0.96610414
x = 0.40, value = 0.99530007
x = 0.50, value = 0.99918606
x = 0.60, value = 0.99530007
x = 0.70, value = 0.96610414
x = 0.80, value = 0.84270046
x = 0.90, value = 0.52049972
x = 1.00, value = 0.00000000
mpirun -np 1 ./sol 0.0001 0.0002 0.0005 500000  19.58s user 0.05s system 99% cpu 19.742 total
```

</details>

---

<details><summary>2 процесса</summary>

```
$ time mpirun -np 2 ./sol 0.0001 0.0002 0.0005 500000
k = 0.0001, tau = 0.0002, h = 0.0005, steps = 500000, points = 2001
k * tau / h^2 = 0.08
x = 0.00, value = 0.00000000
x = 0.10, value = 0.52049972
x = 0.20, value = 0.84270046
x = 0.30, value = 0.96610414
x = 0.40, value = 0.99530007
x = 0.50, value = 0.99918606
x = 0.60, value = 0.99530007
x = 0.70, value = 0.96610414
x = 0.80, value = 0.84270046
x = 0.90, value = 0.52049972
x = 1.00, value = 0.00000000
mpirun -np 2 ./sol 0.0001 0.0002 0.0005 500000  21.91s user 0.07s system 197% cpu 11.122 total
```

</details>

---

<details><summary>4 процесса</summary>

```
$ time mpirun -np 4 ./sol 0.0001 0.0002 0.0005 500000
k = 0.0001, tau = 0.0002, h = 0.0005, steps = 500000, points = 2001
k * tau / h^2 = 0.08
x = 0.00, value = 0.00000000
x = 0.10, value = 0.52049972
x = 0.20, value = 0.84270046
x = 0.30, value = 0.96610414
x = 0.40, value = 0.99530007
x = 0.50, value = 0.99918606
x = 0.60, value = 0.99530007
x = 0.70, value = 0.96610414
x = 0.80, value = 0.84270046
x = 0.90, value = 0.52049972
x = 1.00, value = 0.00000000
mpirun -np 4 ./sol 0.0001 0.0002 0.0005 500000  26.80s user 0.07s system 391% cpu 6.865 total
```

</details>

---

<details><summary>8 процессов</summary>

```
$ time mpirun -np 8 ./sol 0.0001 0.0002 0.0005 500000
k = 0.0001, tau = 0.0002, h = 0.0005, steps = 500000, points = 2001
k * tau / h^2 = 0.08
x = 0.00, value = 0.00000000
x = 0.10, value = 0.52049972
x = 0.20, value = 0.84270046
x = 0.30, value = 0.96610414
x = 0.40, value = 0.99530007
x = 0.50, value = 0.99918606
x = 0.60, value = 0.99530007
x = 0.70, value = 0.96610414
x = 0.80, value = 0.84270046
x = 0.90, value = 0.52049972
x = 1.00, value = 0.00000000
mpirun -np 8 ./sol 0.0001 0.0002 0.0005 500000  35.68s user 0.14s system 773% cpu 4.634 total
```

</details>

---

<details><summary>16 процессов</summary>

```
$ time mpirun -np 16 ./sol 0.0001 0.0002 0.0005 500000
k = 0.0001, tau = 0.0002, h = 0.0005, steps = 500000, points = 2001
k * tau / h^2 = 0.08
x = 0.00, value = 0.00000000
x = 0.10, value = 0.52049972
x = 0.20, value = 0.84270046
x = 0.30, value = 0.96610414
x = 0.40, value = 0.99530007
x = 0.50, value = 0.99918606
x = 0.60, value = 0.99530007
x = 0.70, value = 0.96610414
x = 0.80, value = 0.84270046
x = 0.90, value = 0.52049972
x = 1.00, value = 0.00000000
mpirun -np 16 ./sol 0.0001 0.0002 0.0005 500000  52.15s user 0.27s system 1507% cpu 3.477 total
```

</details>

### 10000 точек

<details><summary>1 процесс</summary>

```
$ time mpirun -np 1 ./sol 0.00001 0.0002 0.0001 500000
k = 1e-05, tau = 0.0002, h = 0.0001, steps = 500000, points = 10001
k * tau / h^2 = 0.2
x = 0.00, value = 0.00000000
x = 0.10, value = 0.97465263
x = 0.20, value = 0.99999226
x = 0.30, value = 1.00000000
x = 0.40, value = 1.00000000
x = 0.50, value = 1.00000000
x = 0.60, value = 1.00000000
x = 0.70, value = 1.00000000
x = 0.80, value = 0.99999226
x = 0.90, value = 0.97465263
x = 1.00, value = 0.00000000
mpirun -np 1 ./sol 0.00001 0.0002 0.0001 500000  96.98s user 0.04s system 99% cpu 1:37.18 total
```

</details>

---

<details><summary>2 процесса</summary>

```
$ time mpirun -np 2 ./sol 0.00001 0.0002 0.0001 500000
k = 1e-05, tau = 0.0002, h = 0.0001, steps = 500000, points = 10001
k * tau / h^2 = 0.2
x = 0.00, value = 0.00000000
x = 0.10, value = 0.97465263
x = 0.20, value = 0.99999226
x = 0.30, value = 1.00000000
x = 0.40, value = 1.00000000
x = 0.50, value = 1.00000000
x = 0.60, value = 1.00000000
x = 0.70, value = 1.00000000
x = 0.80, value = 0.99999226
x = 0.90, value = 0.97465263
x = 1.00, value = 0.00000000
mpirun -np 2 ./sol 0.00001 0.0002 0.0001 500000  103.12s user 0.08s system 199% cpu 51.754 total
```

</details>

---

<details><summary>4 процесса</summary>

```
$ time mpirun -np 4 ./sol 0.00001 0.0002 0.0001 500000
k = 1e-05, tau = 0.0002, h = 0.0001, steps = 500000, points = 10001
k * tau / h^2 = 0.2
x = 0.00, value = 0.00000000
x = 0.10, value = 0.97465263
x = 0.20, value = 0.99999226
x = 0.30, value = 1.00000000
x = 0.40, value = 1.00000000
x = 0.50, value = 1.00000000
x = 0.60, value = 1.00000000
x = 0.70, value = 1.00000000
x = 0.80, value = 0.99999226
x = 0.90, value = 0.97465263
x = 1.00, value = 0.00000000
mpirun -np 4 ./sol 0.00001 0.0002 0.0001 500000  120.64s user 0.75s system 375% cpu 32.308 total
```

</details>

---

<details><summary>8 процессов</summary>

```
$ time mpirun -np 8 ./sol 0.00001 0.0002 0.0001 500000
k = 1e-05, tau = 0.0002, h = 0.0001, steps = 500000, points = 10001
k * tau / h^2 = 0.2
x = 0.00, value = 0.00000000
x = 0.10, value = 0.97465263
x = 0.20, value = 0.99999226
x = 0.30, value = 1.00000000
x = 0.40, value = 1.00000000
x = 0.50, value = 1.00000000
x = 0.60, value = 1.00000000
x = 0.70, value = 1.00000000
x = 0.80, value = 0.99999226
x = 0.90, value = 0.97465263
x = 1.00, value = 0.00000000
mpirun -np 8 ./sol 0.00001 0.0002 0.0001 500000  136.32s user 0.59s system 769% cpu 17.785 total
```

</details>

---

<details><summary>16 процессов</summary>

```
$ time mpirun -np 16 ./sol 0.00001 0.0002 0.0001 500000
k = 1e-05, tau = 0.0002, h = 0.0001, steps = 500000, points = 10001
k * tau / h^2 = 0.2
x = 0.00, value = 0.00000000
x = 0.10, value = 0.97465263
x = 0.20, value = 0.99999226
x = 0.30, value = 1.00000000
x = 0.40, value = 1.00000000
x = 0.50, value = 1.00000000
x = 0.60, value = 1.00000000
x = 0.70, value = 1.00000000
x = 0.80, value = 0.99999226
x = 0.90, value = 0.97465263
x = 1.00, value = 0.00000000
mpirun -np 16 ./sol 0.00001 0.0002 0.0001 500000  166.59s user 1.03s system 1545% cpu 10.847 total
```

</details>

### 50000 точек

<details><summary>1 процесс</summary>

```
$ time mpirun -np 1 ./sol 0.000001 0.0002 0.000020 500000
k = 1e-06, tau = 0.0002, h = 2e-05, steps = 500000, points = 50000
k * tau / h^2 = 0.5
x = 0.00, value = 0.00000000
x = 0.10, value = 1.00000000
x = 0.20, value = 1.00000000
x = 0.30, value = 1.00000000
x = 0.40, value = 1.00000000
x = 0.50, value = 1.00000000
x = 0.60, value = 1.00000000
x = 0.70, value = 1.00000000
x = 0.80, value = 1.00000000
x = 0.90, value = 1.00000000
x = 1.00, value = 0.00000000
mpirun -np 1 ./sol 0.000001 0.0002 0.000020 500000  516.70s user 7.04s system 92% cpu 9:25.55 total
```

</details>

---

<details><summary>2 процесса</summary>

```
$ time mpirun -np 2 ./sol 0.000001 0.0002 0.000020 500000
k = 1e-06, tau = 0.0002, h = 2e-05, steps = 500000, points = 50000
k * tau / h^2 = 0.5
x = 0.00, value = 0.00000000
x = 0.10, value = 1.00000000
x = 0.20, value = 1.00000000
x = 0.30, value = 1.00000000
x = 0.40, value = 1.00000000
x = 0.50, value = 1.00000000
x = 0.60, value = 1.00000000
x = 0.70, value = 1.00000000
x = 0.80, value = 1.00000000
x = 0.90, value = 1.00000000
x = 1.00, value = 0.00000000
mpirun -np 2 ./sol 0.000001 0.0002 0.000020 500000  539.68s user 5.48s system 184% cpu 4:54.83 total
```

</details>

---

<details><summary>4 процесса</summary>

```
$ time mpirun -np 4 ./sol 0.000001 0.0002 0.000020 500000
k = 1e-06, tau = 0.0002, h = 2e-05, steps = 500000, points = 50000
k * tau / h^2 = 0.5
x = 0.00, value = 0.00000000
x = 0.10, value = 1.00000000
x = 0.20, value = 1.00000000
x = 0.30, value = 1.00000000
x = 0.40, value = 1.00000000
x = 0.50, value = 1.00000000
x = 0.60, value = 1.00000000
x = 0.70, value = 1.00000000
x = 0.80, value = 1.00000000
x = 0.90, value = 1.00000000
x = 1.00, value = 0.00000000
mpirun -np 4 ./sol 0.000001 0.0002 0.000020 500000  555.07s user 3.89s system 377% cpu 2:28.17 total
```

</details>

---

<details><summary>8 процессов</summary>

```
$ time mpirun -np 8 ./sol 0.000001 0.0002 0.000020 500000
k = 1e-06, tau = 0.0002, h = 2e-05, steps = 500000, points = 50000
k * tau / h^2 = 0.5
x = 0.00, value = 0.00000000
x = 0.10, value = 1.00000000
x = 0.20, value = 1.00000000
x = 0.30, value = 1.00000000
x = 0.40, value = 1.00000000
x = 0.50, value = 1.00000000
x = 0.60, value = 1.00000000
x = 0.70, value = 1.00000000
x = 0.80, value = 1.00000000
x = 0.90, value = 1.00000000
x = 1.00, value = 0.00000000
mpirun -np 8 ./sol 0.000001 0.0002 0.000020 500000  620.62s user 1.82s system 774% cpu 1:20.35 total
```

</details>

---

<details><summary>16 процессов</summary>

```
$ time mpirun -np 16 ./sol 0.000001 0.0002 0.000020 500000
k = 1e-06, tau = 0.0002, h = 2e-05, steps = 500000, points = 50000
k * tau / h^2 = 0.5
x = 0.00, value = 0.00000000
x = 0.10, value = 1.00000000
x = 0.20, value = 1.00000000
x = 0.30, value = 1.00000000
x = 0.40, value = 1.00000000
x = 0.50, value = 1.00000000
x = 0.60, value = 1.00000000
x = 0.70, value = 1.00000000
x = 0.80, value = 1.00000000
x = 0.90, value = 1.00000000
x = 1.00, value = 0.00000000
mpirun -np 16 ./sol 0.000001 0.0002 0.000020 500000  742.67s user 1.35s system 1567% cpu 47.476 total
```

</details>

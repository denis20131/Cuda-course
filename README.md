# Первое практическое задание — директория `first_practic`

## Описание файлов

- `begin_code.c` — исходный CPU-код, полученный в сообщении в Telegram. Его необходимо **оптимизировать на GPU** и **сравнить производительность** с CPU-версией.
- `common.hpp` — содержит **общие определения**:
  - Тип данных `real` (`float` или `double`)
  - Макросы `IDX`, `MAX`, `EPSILON`
  - Используется и в CPU, и в GPU реализации

---

## Сборка

```bash
# Собрать обе версии (CPU и GPU)
make all

# Только CPU (OpenMP)
make cpu

# Только GPU (CUDA)
make gpu

# Очистить сборочные артефакты
make clean
```

---

##  Запуск

```bash
# CPU-версия с параметрами по умолчанию (L=900, ITMAX=20)
make run_cpu

# GPU-версия с параметрами по умолчанию
make run_gpu

# Запуск обеих версий и сравнение производительности
make run
```

---

##  Цель

Сравнить производительность исходной CPU-реализации и CUDA-оптимизированной 3D-реализации метода Якоби.






# Второе практическое задание — директория `second_practic`
  ## Вся информация как в первом задании

## Результаты второго задания

# Сравнение производительности:

# CPU результаты:

 IT =    1   EPS =  1.4977753E+01

 IT =    2   EPS =  7.4833148E+00

 IT =    3   EPS =  3.7388765E+00

 IT =    4   EPS =  2.8020717E+00

 IT =    5   EPS =  2.0999896E+00

 IT =    6   EPS =  1.6321086E+00

 IT =    7   EPS =  1.3979074E+00

 IT =    8   EPS =  1.2004305E+00

 IT =    9   EPS =  1.0395964E+00

 IT =   10   EPS =  9.0896725E-01

 ADI Benchmark Completed.

 Size            =  900 x  900 x  900

 Iterations      =                 10

 Time in seconds =               4.73

 Operation type  =   double precision

 END OF ADI Benchmark

# GPU результаты:

Compute Device: Tesla P100-SXM2-16GB

Total GPU Memory: 15.90 GB

Iteration    1, Error =  1.4977753E+01

Iteration    2, Error =  7.4833148E+00

Iteration    3, Error =  3.7388765E+00

Iteration    4, Error =  2.8020717E+00

Iteration    5, Error =  2.0999896E+00

Iteration    6, Error =  1.6321086E+00

Iteration    7, Error =  1.3979074E+00

Iteration    8, Error =  1.2004305E+00

Iteration    9, Error =  1.0395964E+00

Iteration   10, Error =  9.0896725E-01

Computation Complete

Grid Dimensions: 900 x 900 x 900

Total Iterations: 10

Time in seconds =  3.299 seconds

Memory Usage: 11123.66 MB


# Ускорение на GPU: в 1.43 раз

# CPU время: 4.73 сек | GPU время: 3.30 сек

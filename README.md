# ЛР4. Параллельное программирование для графического процессора в среде NVidia CUDA.

Должны быть разработаны и последовательный и параллельный вариант программы решения задачи. Не должна требоваться перекомпиляция для запуска программы/программ для различных вариантов исходных данных. Должно быть реализовано программное сравнение результатов последовательных и параллельных вычислений на совпадение и измерены временные затраты на каждый вариант для различных степеней распараллеливания и, возможно, с разными входными данными.
Необходимо добиться того, чтобы параллельный вариант работал быстрее последовательного. Для каждой задачи должны быть проведены несколько вычислительных экспериментов (с различной степенью параллелизма или различными входными данными, …), их результаты должны быть отображены в расчетно-графической работе в форме графиков или таблиц с объяснением характера зависимостей времени вычислений от варьируемых параметров.
Любые исходные данные должны задаваться в командной строке запуска программы и, возможно, считываться из указанного в ней файла (файлов), т.е. они не должны задаваться/определяться как константы в разрабатываемой программе.
Объемные результаты вычислений должны выводиться в файл/файлы. Небольшие по объему результаты могут выводиться на консоль. Выводиться должны все данные, необходимые для проверки полученного результата.
Каждая параллельная программа должна автоматически настраиваться на максимально доступное (или задаваемое при ее запуске) количество потоков/узлов/ядер/ветвей.

**Задача 13. Найти разреженную треугольную подматрицу заданного размера с минимальной суммой элементов в заданной матрице (разреженной называется подматрица, в которой индексы соседних элементов могут различаться более, чем на единицу, например, треугольная подматрица размерностью 4 строки на 4 столбца может описываться, например, такой совокупностью пар индексов: {{1:3},{2:3, 2:5}, {5:3, 5:5, 5:7}, {8:3, 8:5, 8:7, 8:8}}; в этих парах первое число – это индекс строки охватывающей матрицы, второе – это индекс ее столбца; элементов выше главной диагонали нет в отличие от обычного понимания треугольной матрицы)**

Скомпилировать последовательную программу:
`gcc -std=c99 -fopenmp -lm task.c -o ../build/task.out`
Запустить последовательную программу с датасетом:
`../build/task.out ../datasets/data1 ../logs/log.txt`
Скомпилировать параллельную программу:
`make`
Запустить параллельную программу с датасетом:
`../build/task_parallel ../datasets/data1`




Зависимость времени выполнения от размера треугольной подматрицы при размере исходной матрицы 20:
![image](https://github.com/AnasDol/parallel_cuda/assets/51968282/67f44602-9a78-4015-9722-7c7a11af10ec)
Природа графика обусловлена тем, что наибольшая вычислительная сложность наблюдается при размере треугольной матрицы равном половине размера исходной.





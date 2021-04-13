import cv2
import numpy as np
from matplotlib import pyplot as plt
from imageio import imread
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import binary_opening, binary_closing
from skimage.color import label2rgb
from skimage.measure import find_contours, label, regionprops
from scipy.ndimage.morphology import binary_fill_holes

def check(Width_A, Height_A, Widht_B, Height_B): #Сравнение результатов
    print("Check:")
    if Width_A >= Widht_B:
        print("Width A >= width B")
        rW = 1
    else:
        print("WARNING: Weidth A < weidth B")
        rW = 0
    if Height_A >= Height_B:
        print("Height A >= height B")
        rH = 1
    else:
        print("WARNING: Height A < height B")
        rH = 0
    print("\n")
    return rW and rH

def check_result(WIDTH_A, HEIGHT_A, WIDTH_B, HEIGHT_B):
    if WIDTH_A != 0 and HEIGHT_A != 0 and WIDTH_B != 0 and HEIGHT_B != 0:
        res = check(WIDTH_A, HEIGHT_A, WIDTH_B, HEIGHT_B)
    else:
        print("ERROR: this image can't be processed")
        res = 0
    if res == 0:
        print("False")
    else:
        print("True")
    print("\n")
    return res


def calculate_A(contours_pink, mask_pink):  # Вычисления для большой коробки (Объект А)
    WIDTH_A, HEIGHT_A = 0, 0
    (Width_A, Height_A), (max_x_A, max_y_A, min_x_A, min_y_A) = find_size(contours_pink, mask_pink, 2)

    if len(Width_A) < 2 or len(Height_A) < 2 or len(max_x_A) < 2 or len(min_y_A) < 2:
        print("Can't find contours in the image")
    else:
        WIDTH_A = abs(max_x_A[1][1] - min_y_A[0][1])  # по х
        if abs(Height_A[0]) > abs(Height_A[1]):
            HEIGHT_A = abs(Height_A[0])
        else:
            HEIGHT_A = abs(Height_A[1])

        print("WIDTH_A = ", WIDTH_A)
        print("HEIGHT_A = ", HEIGHT_A)

    return WIDTH_A, HEIGHT_A


def download_pink(name_file):
    img_pink = cv2.imread(name_file)

    PINK_MIN = np.array([145, 65, 65], np.uint8)
    PINK_MAX = np.array([179, 255, 255], np.uint8)

    hsv_img = cv2.cvtColor(img_pink, cv2.COLOR_BGR2HSV)

    pink_markers = cv2.inRange(hsv_img, PINK_MIN, PINK_MAX)

    mask_pink = binary_opening(pink_markers, selem=np.ones((10, 10)))

    # plt.imshow(mask_pink)

    # Поиск контуров
    contours_pink = find_contours(mask_pink, level=0.8)

    # Отображение
    # fig, ax = plt.subplots()
    # ax.imshow(pink_markers, cmap=plt.cm.gray)

    # for contour in contours_pink:
    #    ax.plot(contour[:, 1], contour[:, 0], linewidth=1)

    # plt.show()

    return contours_pink, mask_pink

from matplotlib.patches import Rectangle

def calculate_B(contours_img, mask_img): #Вычисления для маленькой коробки (Объекта В)
    WIDTH_B, HEIGHT_B = 0, 0
    (Width_B, Height_B), (max_x_B, max_y_B, min_x_B, min_y_B)= find_size(contours_img, mask_img, 1)

    if len(Width_B) > 0 and len(Height_B) > 0:
        WIDTH_B = abs(Width_B[0])
        HEIGHT_B = abs(Height_B[0])
        print("WIDTH_B = ", WIDTH_B)
        print("HEIGHT_B = ", HEIGHT_B)
        #print(max_x_B)
        #print(max_y_B)
        #print(min_x_B)
        #print(min_y_B)
    else:
        print("Can't find contours in the image")

    return WIDTH_B, HEIGHT_B


def find_size(contours, mask_draw, number):  # Нахождение ширины, длины и минимумом и максимумов по х и у
    index_1, index_2 = find_area(mask_draw, number)

    WIDTH, HEIGHT, MAX_X, MAX_Y, MIN_X, MIN_Y = [], [], [], [], [], []

    index = []
    index.append(index_1)

    if index_2 != -1:  # Если нет второго контура, то пропускаем его соответственно
        index.append(index_2)

    for ind in index:  # Бежим по контурам
        contour = contours[ind]
        max_x, max_y, min_x, min_y = find_min_max(contour)

        # print("max_x: blue = ", max_x)
        # print("min_x: yellow = ", min_x)
        # print("max_y: red = ", max_y)
        # print("min_y: green = ", min_y)

        # Отображение найденных точек на изображении
        # fig, ax = plt.subplots()
        # ax.imshow(mask_draw, cmap=plt.cm.gray)
        # for contour in contours:
        #    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

        # ax.plot(max_x[1], max_x[0], 'bo') #max_x --> самая правая
        # ax.plot(min_x[1], min_x[0], 'yo') #min_x --> самая левая
        # ax.plot(max_y[1], max_y[0], 'ro') #max_y --> самая верхняя
        # ax.plot(min_y[1], min_y[0], 'go') #min_y --> самая нижнаяя

        # plt.show()

        # Вычисление ширины и длины
        if number == 1:
            height = max_x[0] - max_y[0]  # blue - red
        else:
            height = min_y[0] - min_x[0]  # green - yellow
        width = max_x[1] - min_y[1]  # blue - green

        # Добавляем полученные результаты в нужные списки
        WIDTH.append(width)
        HEIGHT.append(height)
        MAX_X.append(max_x)
        MAX_Y.append(max_y)
        MIN_X.append(min_x)
        MIN_Y.append(min_y)

    return (WIDTH, HEIGHT), (MAX_X, MAX_Y, MIN_X, MIN_Y)

def find_min_max(contour): #Вычисление минимумом и максимумов по х и у
    max_x = contour[0]
    max_y = contour[0]
    min_x = contour[0]
    min_y= contour[0]
    for i in range(len(contour)):
        pair = contour[i]
        if max_x[1] < pair[1]:
            max_x = pair
        if min_x[1] > pair[1]:
            min_x = pair
        if min_y[0] < pair[0]:
            min_y = pair
        if max_y[0] > pair[0]:
            max_y = pair
    return max_x, max_y, min_x, min_y


def find_area(mask, number):  # Поиск индексов двух больших контуров на изображении
    largest_area = -1
    largest_contour_index = -1
    largest_area_2 = -1
    largest_contour_index_2 = -1
    labels = label(mask)
    props = regionprops(labels)
    areas = [prop.area for prop in props]

    for i in range(len(areas)):
        if largest_area < areas[i]:
            largest_area = areas[i]
            largest_contour_index = i

    if number == 2:
        for i in range(len(areas)):
            if i != largest_contour_index:
                if largest_area_2 < areas[i]:
                    largest_area_2 = areas[i]
                    largest_contour_index_2 = i
    else:
        largest_area_2 = -1
        largest_contour_index_2 = -1

    return largest_contour_index, largest_contour_index_2


def download(name_file):
    img = imread(name_file)
    img_gray = rgb2gray(img)

    binary_img = binary_closing(canny(img_gray, sigma=2), selem=np.ones((6, 6)))  # 1.5;2
    img_segment = binary_fill_holes(binary_img)
    mask_img = binary_opening(img_segment, selem=np.ones((40, 40)))  # 16;25

    plt.imshow(img)
    # plt.imshow(label2rgb(mask_img, image=img_gray))

    # Поиск контуров
    contours_img = find_contours(mask_img, level=0.8)

    # Отображение (слишком громоздко при запуске на всем датасете, поэтому закомментировано)
    # fig, ax = plt.subplots()
    # ax.imshow(mask_img, cmap=plt.cm.gray)

    # for contour in contours_img:
    #    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    plt.show()

    return contours_img, mask_img


def main_function(name_file):
    WIDTH_A, WIDTH_B, HEIGHT_A, HEIGHT_B = 0, 0, 0, 0
    contours_img, mask_img = download(name_file)  # Ищем контуры маленькой коробочки
    WIDTH_B, HEIGHT_B = calculate_B(contours_img, mask_img)  # Вычисление итогой ширины и длины маленькой коробки
    contours_pink, mask_pink = download_pink(name_file)  # Ищем контуры розовых меток на большой коробке
    WIDTH_A, HEIGHT_A = calculate_A(contours_pink, mask_pink)  # Вычисление итоговой ширины и длины большой коробки
    res = check_result(WIDTH_A, HEIGHT_A, WIDTH_B, HEIGHT_B)  # Сравнение результатов коробок
    return res


# Работа алгоритма на всем датасете
result_yes = []
result_no = []
pair = []
count_no = 0
count_yes = 0
all_count_yes = 8
all_count_no = 18

# Пробегаемся по датасету с меткой Нет (т.е. коробочка не помещается в большую коробку), высчитываем и запоминаем результат
for i in range(1, all_count_no + 1):
    img_name = "data" + str(i) + ".jpg"
    print("Check ", img_name)
    res_no = main_function("dataset\\no\\" + img_name)
    print("res_no = ", res_no)
    pair.append(i)
    pair.append(res_no)
    result_no.append(pair)
    pair = []

# Пробегаемся по датасету с меткой Да (т.е. коробочка помещается в большую коробку), высчитываем и запоминаем результат
for i in range(1, all_count_yes + 1):
    img_name = "data" + str(i) + ".jpg"
    print("Check ", img_name)
    res_yes = main_function("dataset\\yes\\" + img_name)
    print("res_yes = ", res_yes)
    pair.append(i)
    pair.append(res_yes)
    result_yes.append(pair)
    pair = []

# Правильно ли определили, что коробка НЕ помещается
for i in range(len(result_no)):
    if result_no[i][1] == 0:
        print("Result is true! (False)")
        count_yes = count_yes + 1
    else:
        print("Result is false! (False)")
        count_no = count_no + 1
    print("\n")

# Правильно ли определили, что коробка помещается
for i in range(len(result_yes)):
    if result_yes[i][1] == 1:
        print("Result is true! (True)")
        count_yes = count_yes + 1
    else:
        print("Result is false! (True)")
        count_no = count_no + 1
    print("\n")

# Точность работы алгоритма
print("count_yes = ", count_yes)
print("count_no = ", count_no)
print("accuracy = ", count_yes / (all_count_yes + all_count_no))

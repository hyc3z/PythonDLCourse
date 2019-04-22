import pyocr
import importlib
import sys
import time
import re
importlib.reload(sys)
time1 = time.time()
# print("初始时间为：",time1)

import os.path


if __name__ == '__main__':
    for subdir_1 in os.listdir('.'):
        if os.path.isdir(subdir_1):
            for file in os.listdir(subdir_1):
                print('pdfx -d ../crawl/ "'+os.path.join('.', subdir_1, file)+'"')
                os.system('pdfx -d ../crawl/ "'+os.path.join('.', subdir_1, file)+'"')

    time2 = time.time()
    print("总共消耗时间为:", time2 - time1)
import logging
import os
import time


def getLogger(home_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    # FileHandler
    # work_dir = os.path.join('loggings', time.strftime("%Y-%m-%d-%H.%M", time.localtime()))  # 日志文件写入目录
    # if not os.path.exists(work_dir):
    #     os.makedirs(work_dir)
    fHandler = logging.FileHandler(home_dir + '/log.txt', mode='w')
    fHandler.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    fHandler.setFormatter(formatter)  # 定义handler的输出格式
    logger.addHandler(fHandler)  # 将logger添加到handler里面

    return logger
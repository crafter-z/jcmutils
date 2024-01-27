"""此模块负责JCMsuite工程项目的解算
"""
import os
import shutil
import datetime
import cv2
import jcmwave
import numpy as np
from .logger import logger


# 包含了解算器的全部相关代码
class solver:
    def __init__(self, jcmp_path, database_path, keys):
        """初始化solver
        :param  jcmp_path: project.jcmp所在的路径
        :param  database_path: resultbag.db所在的路径，使用不同的名字可改用不同的db
        :param  keys: 在jcmpt或jcmt中所需要替换的全部key，以列表形式给出
        :return nothing
        """
        # 初始化成员变量
        self.jcmp_path = jcmp_path
        self.keys = keys
        if os.path.isabs(database_path):
            abs_resultbag_dir = database_path
        else:
            abs_resultbag_dir = os.path.join(os.getcwd(), database_path)
        if not os.path.exists(os.path.dirname(database_path)):
            os.makedirs(os.path.dirname(database_path))
        self.resultbag = jcmwave.Resultbag(abs_resultbag_dir)
        self.has_inited = True
        logger.info("solver inited")
        logger.debug(
            f"solver parameters:jcmp_path-{jcmp_path};database_path-{abs_resultbag_dir}"
        )

    def solve(self):
        """解算项目，核心函数
        无需参数,信息已在类初始化时定义过
        无返回值，执行结束后所有信息都将已存在于resultbag.db中
        """
        self.__check_logger()

        # 初始化变量
        job_ids = []
        waiting_keys = self.keys
        no_error = False

        # 当存在有报错的项目或首次执行时进入循环
        while not no_error:
            # 开始计算
            last_time = self.__start_timing()
            for key in waiting_keys:
                job_id = jcmwave.solve(
                    self.jcmp_path, keys=key, temporary=True, resultbag=self.resultbag
                )
                job_ids.append(job_id)
            logger.info("solve queue added done! start solving")
            jcmwave.daemon.wait(job_ids, resultbag=self.resultbag)
            cost_time = self.__count_time(last_time)
            logger.info(
                f"solver program completed!,cost{cost_time} analysing results..."
            )
            no_error = True

            # 提取错误信息，如果是OOM错误则加入队列重新计算，否则直接爆Exception
            backup_keys = []
            for key in waiting_keys:
                jcm_log = self.resultbag.get_log(key)
                if jcm_log["ExitCode"] == 0:
                    logger.debug("the key shown below was with no error")
                    logger.debug(f"the key is : {key}")
                    continue
                if "memory" in jcm_log["Log"]["Error"]:
                    logger.warning("Memory Limit Exceeded!! trying to solve it again")
                    logger.warning(f"the key is : {key}")
                    no_error = False
                    self.resultbag.remove_result(key)
                    backup_keys.append(key)
                else:
                    logger.critical(
                        "FATAL ERROR! Unknown error occoured while solving projects !"
                    )
                    logger.critical('Error Message : "%s"', jcm_log["Log"]["Error"])
                    logger.critical(f"the key is : {key}")
                    raise Exception(
                        "Unknown error occoured while solving projects! please check the log file"
                    )

            # 如果出现了oom错误，替换队列，再次计算
            if not no_error:
                waiting_keys = backup_keys

        logger.info("analyse complete ! No error report ! solve mission done!!")

    def show_image(self, key, num_of_result, is_light_intense=False, vmax=None):
        """显示resultbag中指定的key对应的第num_of_result中的电场矢量对应的光强图
        需要第num_of_result中的结果是ExportFields出来的ElectricFieldStrength才能正常显示，否则报错
        :param  is_light_intense: 如果为True，输出光强图，否则是电场强度图
        :param  vmax: 以0-vmax的(场强/光强)来对应0-235的像素值
        """
        self.__check_logger()
        self.check_result(key)

        # 开始提取
        result = self.resultbag.get_result(key)
        field = (
            (
                result[num_of_result]["field"][0].conj()
                * result[num_of_result]["field"][0]
            )
            .sum(axis=2)
            .real
        )
        if is_light_intense:
            field = np.power(field, 2)
        vmaxa = np.max(field) if vmax is None else vmax
        field = (field / vmaxa) * 235
        field = np.rot90(field)
        cv2.imshow("image", field)

    def get_result(self, key):
        """返回resultbag中该key对应的result"""
        self.__check_logger()
        self.check_result(key)
        return self.resultbag.get_result(key)

    def save_image(
        self, target_directory, key, num_of_result, is_light_intense=False, vmax=None
    ):
        """保存一张图像，大部分参数及意义与show_image函数相同
        :param  target_directory: 图像将被保存至traget_directory/output.jpg
        """
        self.__check_logger()
        self.check_result(key)

        # 开始提取
        result = self.resultbag.get_result(key)
        field = (
            (
                result[num_of_result]["field"][0].conj()
                * result[num_of_result]["field"][0]
            )
            .sum(axis=2)
            .real
        )
        if is_light_intense:
            field = np.power(field, 2)
        if not os.path.exists(target_directory):
            logger.info("target directory dosen't exist,creating...")
            os.makedirs(target_directory)
        vmaxa = np.max(field) if vmax is None else vmax
        field = (field / vmaxa) * 235
        field = np.rot90(field)
        cv2.imwrite(target_directory.rstrip("/") + "output.jpg", field)
        logger.info("target image saved successfully")

    def save_all_image(
        self,
        num_of_result,
        target_directory,
        is_light_intense=False,
        is_symmetry=False,
        vmax=None,
    ):
        """保存resultbag中的所有图像至目标路径,同时输出一个所有图像加和形成的total_results.jpg
        大部分参数与save_image相同
        :param  is_symmetry: 是否是镜像，需要生成的科勒照明光同样启用了is_symmetry
        """
        self.__check_logger()
        self.check_result(self.keys[0])

        # 计时
        last_time = self.__start_timing()

        # 开始提取
        # 先确定total_result的形状
        temp_result = self.resultbag.get_result(self.keys[0])
        field = (
            (
                temp_result[num_of_result]["field"][0].conj()
                * temp_result[num_of_result]["field"][0]
            )
            .sum(axis=2)
            .real
        )
        total_results = np.zeros(field.shape)
        logger.debug(f"total_result shape defined as {total_results.shape}")

        # 开始逐个提取结果
        for key in self.keys:
            # 目录检查
            if not os.path.exists(target_directory):
                logger.debug("target directory dosen't exist,creating...")
                os.makedirs(target_directory)
            file_name = (
                target_directory.rstrip("/") + "/" + self.__solve_dict(key) + ".jpg"
            )

            # 获得结果
            result = self.resultbag.get_result(key)
            field = (
                (
                    result[num_of_result]["field"][0].conj()
                    * result[num_of_result]["field"][0]
                )
                .sum(axis=2)
                .real
            )
            if is_light_intense:
                field = np.power(field, 2)

            field = np.rot90(field)  # 这一步是必须的，因为jcm导出的顺序是没翻过来的
            save_field = (field / np.max(field)) * 235
            cv2.imwrite(file_name, save_field)
            logger.debug(f"key {key} successfully saved")
            total_results += field
            if is_symmetry and not (
                key["thetaphi"][0] == 0 and key["thetaphi"][1] == 0
            ):
                field = np.rot90(field, 2)
                total_results += field
                logger.debug("key was rotated for symmetry")

        logger.info(f"printing max value of results:{np.max(total_results)}")
        vmaxa = np.max(total_results) if vmax is None else vmax
        sfield = (total_results / vmaxa) * 235
        file_name = target_directory.rstrip("/") + "/" + "total_result.jpg"
        cv2.imwrite(file_name, sfield)
        cost_time = self.__count_time(last_time)
        logger.info(f"all target image saved completed! cost {cost_time}")

    # 将root_dir目录下面的total_result.jpg移动至目标位置
    def move_total_results(self, root_dir, target_dir):
        self.__check_logger()
        filelist = os.listdir(root_dir)
        for file in filelist:
            if file == "total_result.jpg":
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                shutil.copyfile(
                    os.path.join(root_dir, file),
                    os.path.join(target_dir, os.path.basename(root_dir) + ".jpg"),
                )

    # 检查是否存在目标键值，如无则报错
    def check_result(self, key):
        if not self.resultbag.check_result(key):
            logger.error("get result failed! target key not find in resultbag")
            logger.error(f"the key is : {key}")
            raise Exception(
                f"get result failed! target key not find in resultbag,the key is {key}"
            )

    # 工具函数，处理词典，用处是将保存时的文件名缩短,key中包含的所有参数（浮点数保留两位）连接起来形成文件名。
    def __solve_dict(self, target_dict):
        res = ""
        for key, value in target_dict.items():
            res += key + "-"
            if isinstance(value, list):
                for i in value:
                    if isinstance(i, float):
                        res += "{:.2f}-".format(i)
                    else:
                        res += f"{i}-"
            else:
                res += f"{value}-"
        res.rstrip("-")
        return res

    # 检查是否被以被初始化，若还未被初始化则报错
    def __check_logger(self):
        try:
            if self.has_inited:
                logger.debug("solver class have been inited")
        except NameError:
            print("Error ! please init solver befor using it!!!!!!")
            raise Exception("Please init solver before using it")

    def __start_timing(self):
        return datetime.datetime.now()

    def __count_time(self, last_time):
        return datetime.datetime.now() - last_time

import jcmwave
import numpy as np
import logging
import matplotlib.pyplot as plt


class solver:
    def __init__(self, jcmp_path, database_path, keys, use_log_file, logfile_path="./jcmsolver.log", log_level=logging.DEBUG) -> None:
        # 初始化成员变量
        self.jcmp_path = jcmp_path
        self.keys = keys
        self.logfile_path = logfile_path
        self.resultbag = jcmwave.Resultbag(database_path)
        LOG_FORMAT = "|%(asctime)s - %(levelname)s|->%(message)s"
        DATA_FROMAT = "%Y/%m/%d %H:%M:%S"

        # 初始化log
        if use_log_file:
            logging.basicConfig(format=LOG_FORMAT,
                                datefmt=DATA_FROMAT, level=log_level)
        else:
            logging.basicConfig(
                filename=logfile_path, format=LOG_FORMAT, datefmt=DATA_FROMAT, level=log_level)
        logger = logging.getLogger('solver')
        self.logger = logger
        logger.info("logging started!")
        logger.info("logging parameters are listed as follow:")
        logger.info("jcmp_path = %s", jcmp_path)
        logger.info("database_path = %s", database_path)
        logger.info("logfile_path = %s", logfile_path)
        logger.info("log_level = %d", log_level)
        self.has_inited = True

    def solve(self):
        # 检查是否被以被初始化，若还未被初始化则报错
        try:
            if self.has_inited:
                self.logger.debug("solver class have been inited")
        except NameError:
            print("Error ! please init solver befor using it!!!!!!")
            raise Exception("Please init solver before using it")

        # 初始化变量
        logger = self.logger
        job_ids = []
        waiting_keys = self.keys
        no_error = False

        # 当存在有报错的项目或首次执行时进入循环
        while not no_error:

            # 开始计算
            for key in waiting_keys:
                job_id = jcmwave.solve(
                    self.jcmp_path, keys=key, temporary=True, resultbag=self.resultbag)
                job_ids.append(job_id)
            logger.info("solve queue added done! start solving")
            jcmwave.daemon.wait(job_ids, resultbag=self.resultbag)
            logger.info("solver program completed! analysing results...")
            no_error = True

            # 提取错误信息，如果是mmo错误则加入队列重新计算
            backup_keys = []
            for key in waiting_keys:
                jcm_log = self.resultbag.get_log(key)
                if jcm_log['ExitCode'] == 0:
                    logger.debug("the key shown below was with no error")
                    logger.debug(f"the key is : {key}")
                    continue
                if "memory" in jcm_log['Log']['Error']:
                    logger.warning(
                        "Memory Limit Exceeded!! trying to solve it again")
                    logger.warning(f"the key is : {key}")
                    no_error = False
                    backup_keys.append(key)
                else:
                    logger.critical(
                        "FATAL ERROR! Unknown error occoured while solving projects !")
                    logger.critical("Error Message : \"%s\"",
                                    jcm_log['Log']['Error'])
                    logger.critical(f"the key is : {key}")
                    raise Exception(
                        "Unknown error occoured while solving projects! please check the log file")

            # 如果出现了mmo错误，替换队列，再次计算
            if not no_error:
                waiting_keys = backup_keys

        logger.info("analyse complete ! No error report ! solve mission done!!")

    def show_image(self, key, num_of_result, is_light_intense=False):
        logger = self.logger
        if not self.resultbag.check_result(key):
            logger.error("get result failed! target key not find")
            logger.error(f"the key is : {key}")
            return -1

        # 开始提取
        result = self.resultbag.get_result(key)
        field = (result[num_of_result]['field'][0].conj() *
                 result[num_of_result]['field'][0]).sum(axis=2).real
        if is_light_intense:
            field = np.power(field, 2)
        plt.cla()
        plt.axis('square')
        plt.axis('off')
        plt.pcolormesh(field[num_of_result]['X'], field[num_of_result]['Y'],
                       field, shading='gouraud', cmap='gray')
        plt.show()

    def get_result(self, key):
        return self.resultbag.get_result(key)

    def save_image(self, file_path, key, num_of_result, is_light_intense=False):
        logger = self.logger
        if not self.resultbag.check_result(key):
            logger.error("get result failed! target key not find")
            logger.error(f"the key is : {key}")
            return -1

        # 开始提取
        result = self.resultbag.get_result(key)
        field = (result[num_of_result]['field'][0].conj() *
                 result[num_of_result]['field'][0]).sum(axis=2).real
        if is_light_intense:
            field = np.power(field, 2)
        plt.cla()
        plt.axis('square')
        plt.axis('off')
        plt.pcolormesh(field[num_of_result]['X'], field[num_of_result]['Y'],
                       field, shading='gouraud', cmap='gray')
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)

    def save_all_image(self, num_of_result, target_directory, is_light_intense=False, is_symmetry=False):
        logger = self.logger
        if not self.resultbag.check_result(self.keys[0]):
            logger.error("get result failed! target key not find")
            logger.error(f"the key is : {self.keys[0]}")
            return -1

        # 开始提取
        # 先确定total_result的形状
        temp_result = self.resultbag.get_result(self.keys[0])
        field = (temp_result[num_of_result]['field'][0].conj() *
                 temp_result[num_of_result]['field'][0]).sum(axis=2).real
        total_results = np.zeros(field.shape)
        logger.debug(f"total_result shape defined as {total_results.shape}")

        # 开始逐个提取结果
        for key in self.keys:
            result = self.resultbag.get_result(key)
            field = (result[num_of_result]['field'][0].conj() *
                     result[num_of_result]['field'][0]).sum(axis=2).real
            if is_light_intense:
                field = np.power(field, 2)
            total_results += field
            file_name = target_directory.rstrip(
                '/') + '/' + self.__solve_dict(key) + ".jpg"
            plt.cla()
            plt.axis('square')
            plt.axis('off')
            plt.pcolormesh(field[num_of_result]['X'], field[num_of_result]['Y'],
                           field, shading='gouraud', cmap='gray')
            plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
            logger.debug(f"key {key} successfully saved")
            if is_symmetry:
                field = np.rot90(field, 2)
                total_results += field
                logger.debug("key was rotated for symmetry")

        plt.cla()
        plt.axis('square')
        plt.axis('off')
        plt.pcolormesh(field[num_of_result]['X'], field[num_of_result]['Y'],
                       total_results, shading='gouraud', cmap='gray')
        file_name = target_directory.rstrip('/') + '/' + "total_result.jpg"
        plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
        logger.info("all target image saved completed!")

    def __solve_dict(self, target_dict):
        res = ""
        for key, value in target_dict.items():
            res += key + "-"
            for i in value:
                if isinstance(i, float):
                    res += "{:.2f}".format(i) + "-"
        res.rstrip('-')
        return res

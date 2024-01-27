"""log的相关定义
"""
import logging


class logger_level:
    """log的相关定义
    :logger_level 定义信息的严重等级
    """

    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    NOTSET = 0


class __logger_class:
    def __init__(self) -> None:
        self.__logger = logging.getLogger("root")

    def init_logger(
        self,
        logger_name,
        use_logfile=False,
        logfile_path="jcmlog.log",
        log_format="|%(asctime)s - %(levelname)s|->%(message)s",
        data_format="%Y/%m/%d %H:%M:%S",
        log_level=logger_level.DEBUG,
    ):
        """初始化logger和logger的参数，使用时需手动调用
        :param  logger_name: logger的名字
        :param  use_logfile: 是否将log保存至文件，为True时会将log保存至文件而不是输出至控制台
        :param  logfile_path: use_logfile为True时才有效，将log保存至何处
        :param  log_format: log的格式，如需更改，按照logging模块的log模式来修改
        :param  data_format: 输出的时间的格式，按照logging模块来修改
        :param  log_level: 取logger_level中的某个值，当log比该等级高时，会将该log输出，否则隐藏
        :return 初始化成功后的logger
        """
        self.__logger = logging.getLogger(logger_name)
        if use_logfile:
            fh = logging.FileHandler(
                logfile_path, mode="a", encoding="UTF-8", delay=False
            )
            fh.setFormatter(logging.Formatter(log_format, data_format))
            self.__logger.setLevel(log_level)
            self.__logger.addHandler(fh)
            return self.__logger
        else:
            sh = logging.StreamHandler(stream=None)
            sh.setFormatter(logging.Formatter(log_format, data_format))
            self.__logger.setLevel(log_level)
            self.__logger.addHandler(sh)
            return self.__logger

    def info(self, msg, *args):
        self.__logger.info(msg, *args)

    def debug(self, msg, *args):
        self.__logger.debug(msg, *args)

    def warning(self, msg, *args):
        self.__logger.warning(msg, *args)

    def error(self, msg, *args):
        self.__logger.error(msg, *args)

    def critical(self, msg, *args):
        self.__logger.fatal(msg, *args)


logger = __logger_class()
"""
logger为单例，全局只包含一个logger，不能额外定义logger_class
"""

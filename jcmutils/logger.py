import logging


class logger_level:
    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    NOTSET = 0
class logger_class:
    def __init__(self) -> None:
        self.__logger = logging.getLogger("root")

    def init_logger(self, logger_name,use_logfile=False, logfile_path="jcmlog.log", log_format="|%(asctime)s - %(levelname)s|->%(message)s", data_format="%Y/%m/%d %H:%M:%S", log_level=logger_level.DEBUG):
        self.__logger = logging.getLogger(logger_name)
        if use_logfile:
            fh = logging.FileHandler(
                logfile_path, mode='a', encoding='UTF-8', delay=False)
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

    def info(self,msg,*args):
        self.__logger.info(msg,*args)
    def debug(self,msg,*args):
        self.__logger.debug(msg,*args)
    def warning(self,msg,*args):
        self.__logger.warn(msg,*args)
    def error(self,msg,*args):
        self.__logger.error(msg,*args)
    def critical(self,msg,*args):
        self.__logger.fatal(msg,*args)


logger = logger_class()

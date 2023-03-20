import logging

def init_logger(use_logfile= False, logfile_path= "jcmlog.log",log_format= "|%(asctime)s - %(levelname)s|->%(message)s",data_format= "%Y/%m/%d %H:%M:%S", log_level= logging.DEBUG):
    logger = logging.getLogger("JCM")
    if use_logfile:
        fh = logging.FileHandler(logfile_path,mode='a',encoding='UTF-8',delay=False)
        fh.setFormatter(logging.Formatter(log_format,data_format))
        logger.setLevel(log_level)
        logger.addHandler(fh)
        return logger
    else:
        sh = logging.StreamHandler(stream=None)
        sh.setFormatter(logging.Formatter(log_format,data_format))
        logger.setLevel(log_level)
        logger.addHandler(sh)
        return logger
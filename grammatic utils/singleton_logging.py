import logging
import datetime


# inherent class
class Logger(logging.Logger):
    def __init__(self):
        super().__init__('logger')
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')
        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(filename='log/logfile_' + str(datetime.datetime.now()) + '.log')

        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        self.addHandler(stream_handler)
        self.addHandler(file_handler)


# Singleton class
class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# combine it!
class Log(Logger, metaclass=Singleton):
    pass


def main():

    logger = Log()
    logger.debug('hello world!')


if __name__ == '__main__':
    main()

import os, time, logging, json
from logging import Filter
from logging.handlers import QueueHandler, QueueListener
import logging.config
import logging, traceback

import torch
from torch.multiprocessing import Queue

def create_log_directories(root, env_name):
    """Create log directory.
    :arg        (str)   root        :  Root for log directory
                (str)   env_name    :  Environment name
    :returns    (dict)  result_dir  :  log directory
    """

    directories = {}

    if not os.path.isdir(root): os.mkdir(root)
    directories['base'] = root+"/%s" % time.strftime("%m%d-%H%M"+env_name)

    # For logs of testing and training
    directories['log'] = directories['base']+"/log"
    directories['train'] = directories['log']+"/train"
    directories['test'] = directories['log']+"/test"

    #  For saving the buffer
    directories['buffer'] = directories['base']+"/buffer"

    # For saving networks : policy, model
    directories['network'] = directories['base']+"/network"
    directories['policy'] = directories['network']+"/policy"
    directories['model'] = directories['network']+"/model"
    directories['dnn'] = directories['model']+"/dnn"
    directories['bnn'] = directories['model']+"/bnn"
    directories['rbf'] = directories['model'] + "/rbf"

    for key in directories.keys():
        if not os.path.isdir(directories[key]):
            os.mkdir(directories[key])

    return directories


def load_log_directories(fname):

    directories = {}

    directories['base'] = "./results/" + fname

    # For logs of testing and training
    directories['log'] = directories['base']+"/log"
    directories['train'] = directories['log']+"/train"
    directories['test'] = directories['log']+"/test"

    #  For saving the buffer
    directories['buffer'] = directories['base']+"/buffer"

    # For saving networks : policy, model
    directories['network'] = directories['base']+"/network"
    directories['policy'] = directories['network']+"/policy"
    directories['model'] = directories['network']+"/model"
    directories['dnn'] = directories['model']+"/dnn"
    directories['bnn'] = directories['model']+"/bnn"

    return directories

def create_update_logger(file):
    """Create configuration logger.
    :arg (str) file   : Log file.
    :returns (instance) update_logger   : Update logger.
    """
    with open('./tool/logging.json', 'rt') as f:
        config = json.load(f)
    logging.config.dictConfig(config)
    update_logger = logging.getLogger("monitor")
    timeHandler = logging.FileHandler(file)
    update_logger.addHandler(timeHandler)
    return update_logger

def create_config_logger(sac_config, file):
    """Create configuration logger.
    :arg (instance) airsim_config    : Airsim configuration.
    :arg (instance) sac_config       : SAC configuration.
    :arg (str) file                  : Log file.
    :returns (instance) config_logger   : Configuration logger.
    """
    with open('./tool/logging.json', 'rt') as f:
        config = json.load(f)
    logging.config.dictConfig(config)
    config_logger = logging.getLogger("configuration")
    FileHandler = logging.FileHandler(file)
    config_logger.addHandler(FileHandler)
    # config_logger.info("\n################ AirSim configuration ################\n")
    # keys = list(airsim_config.__dict__.keys())[1:28]
    # for key in keys:
    #     val = airsim_config.__dict__[key]
    #     config_logger.info("- %s: %r", key, val)
    config_logger.info("\n################## SAC configuration #################\n")
    for arg, value in vars(sac_config).items():
        config_logger.info("- %s: %r", arg, value)
    return config_logger

def create_evaluation_logger(file, name):
    """Create evaluation logger.
    :arg (str) file    : Log file.
    :arg (str) name    : Logger identification.
    :returns (instance) eval_logger   : Evaluation logger.
    """
    eval_logger = logging.getLogger("evaluation_{0}".format(name))
    evalHandler = logging.FileHandler(file)
    eval_logger.addHandler(evalHandler)
    return eval_logger

def setup_primary_logging(log_file_path: str, error_log_file_path: str) -> Queue:
    """
    Global logging is setup using this method. In a distributed setup, a multiprocessing queue is setup
    which can be used by the workers to write their log messages. This initializers respective handlers
    to pick messages from the queue and handle them to write to corresponding output buffers.
    Parameters
    ----------
    log_file_path : ``str``, required
        File path to write output log
    error_log_file_path: ``str``, required
        File path to write error log
    Returns
    -------
    log_queue : ``torch.multiprocessing.Queue``
        A log queue to which the log handler listens to. This is used by workers
        in a distributed setup to initialize worker specific log handlers(refer ``setup_worker_logging`` method).
        Messages posted in this queue by the workers are picked up and bubbled up to respective log handlers.
    """
    # Multiprocessing queue to which the workers should log their messages
    log_queue = Queue(-1)

    # Handlers for stream/file logging
    output_file_log_handler = logging.FileHandler(filename=str(log_file_path))
    error_file_log_handler = logging.FileHandler(filename=str(error_log_file_path))

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    output_file_log_handler.setFormatter(formatter)
    error_file_log_handler.setFormatter(formatter)

    output_file_log_handler.setLevel(logging.INFO)
    error_file_log_handler.setLevel(logging.ERROR)

    # This listener listens to the `log_queue` and pushes the messages to the list of
    # handlers specified.
    listener = QueueListener(log_queue, output_file_log_handler, error_file_log_handler, respect_handler_level=True)

    listener.start()

    return log_queue

class WorkerLogFilter(Filter):
    def __init__(self, rank=-1):
        super().__init__()
        self._rank = rank

    def filter(self, record):
        if self._rank != -1:
            record.msg = f"Rank {self._rank} | {record.msg}"
        return True

def setup_worker_logging(rank: int, log_queue: Queue):
    """
    Method to initialize worker's logging in a distributed setup. The worker processes
    always write their logs to the `log_queue`. Messages in this queue in turn gets picked
    by parent's `QueueListener` and pushes them to respective file/stream log handlers.
    Parameters
    ----------
    rank : ``int``, required
        Rank of the worker
    log_queue: ``Queue``, required
        The common log queue to which the workers
    Returns
    -------
    features : ``np.ndarray``
        The corresponding log power spectrogram.
    """
    queue_handler = QueueHandler(log_queue)

    # Add a filter that modifies the message to put the
    # rank in the log format
    worker_filter = WorkerLogFilter(rank)
    queue_handler.addFilter(worker_filter)
    queue_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()

    root_logger = logging.getLogger()
    root_logger.addHandler(queue_handler)
    root_logger.addHandler(stream_handler)

    # Default logger level is WARNING, hence the change. Otherwise, any worker logs
    # are not going to get bubbled up to the parent's logger handlers from where the
    # actual logs are written to the output
    root_logger.setLevel(logging.INFO)

def log_error(fname):
    logging.basicConfig(filename=fname, level=logging.ERROR)
    logging.error(traceback.format_exc())


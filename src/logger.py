import os
import tempfile
import pickle
import json

_clearm = True
try:
    from clearml import Task
    from clearml import Logger
except:
    _clearm = False


class GlobalLogger:
    _loggers = None
    _it = 0

    @classmethod
    def tick(cls):
        cls._it += 1

    @classmethod
    def add_logger(cls, logger):
        if cls._loggers is None: cls._loggers = []
        cls._loggers.append(logger)
    
    @classmethod
    def log(cls, key, value):
        if cls._loggers is None: return
        for logger in cls._loggers:
            logger.log(cls._it, key, value)

    @classmethod
    def log_media(cls, key, media):
        if cls._loggers is None: return
        for logger in cls._loggers:
            if hasattr(logger, "log_media"):
                logger.log_media(cls._it, key, media)

    @classmethod
    def log_object(cls, key, obj, format="pickle"):
        if cls._loggers is None: return
        for logger in cls._loggers:
            if hasattr(logger, "log_object"):
                logger.log_object(cls._it, key, obj, format)

class StdoutLogger:
    def log(self, it, key, value):
        print(f"SCALAR (it={it}) {key}: {value}")
    def log_media(self, it, key, media):
        print(f"MEDIA (it={it}) {key}: {media}")
    def log_object(self, it, key, object):
        print(f"OBJECT (it={it}) {key}: {object}")

class ClearMLLogger:
    def __init__(self, project, task, output_uri, media_uri, seed, tags=[], newtask=False):
        assert _clearm, "Please install clearml"

        Task.add_requirements("./requirements.txt")
        Task.set_random_seed(seed)
        
        task = Task.init(
            project_name=project,
            task_name=task,
            tags=tags,
            output_uri=output_uri,
            reuse_last_task_id=not newtask,
            auto_connect_frameworks=dict(matplotlib=False)
        )
        Logger.current_logger().set_default_upload_destination(media_uri)

        queue = os.environ.get("CLEARML_QUEUE", False) 
        if queue:
            task.close()
            task.reset()
            Task.enqueue(task, queue_name=queue)
            print(f"Task enqueued on {queue}")
            exit()

    def log(self, it, key, value):
        Task.current_task().get_logger().report_scalar(
            title=key,
            series=key,
            value=value,
            iteration=it
        )
    
    def log_media(self, it, key, value):
        Task.current_task().get_logger().report_media(
            title=key,
            series=key,
            local_path=value,
            iteration=it,
            delete_after_upload=True
        )
    
    def log_object(self, it, key, object, format="pickle"):
        assert format in ["pickle", "json"], "Only pickle and json format are supported"
        if format == "json":
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
            with open(temp.name, "w") as f:
                json.dump(object, f)
        elif format == "pickle":
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
            with open(temp.name, "wb") as f:
                pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        Task.current_task().get_logger().report_media(
            title=key,
            series=key,
            local_path=temp.name,
            iteration=it,
            delete_after_upload=True
        )
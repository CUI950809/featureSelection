from utility.conf import time


LSFSCWTime = []
LSFSGetWTime = []
LSFSTime = []
LSFSresult = []
LSFSObejectV = []
LSFSWObejectV = []


def timeit(time_list):
    def _deco(fun):
        def __deco(*args, **kwargs):
            __deco.__name__ = fun.__name__
            import time
            s = time.clock()
            result = fun(*args, **kwargs)
            dual = time.clock() - s
            time_list.append(dual)
            return result
        return __deco
    return _deco


def get_object_value(obejectVList):
    def _deco(fun):
        import copy
        def __deco(*args, **kwargs):
            __deco.__name__ = fun.__name__
            result = fun(*args, **kwargs)
            obejectVList.append(copy.deepcopy(result))
            return result
        return __deco
    return _deco


def reset_lsfs_global_value(fun):
    def _deco(*args, **kwargs):
        _deco.__name__ = fun.__name__
        global LSFSGetWTime, LSFSCWTime, LSFSWObejectV, LSFSObejectV
        LSFSGetWTime = []
        LSFSCWTime = []
        LSFSWObejectV = []
        LSFSObejectV = []
        result = fun(*args, **kwargs)
        return result
    return _deco


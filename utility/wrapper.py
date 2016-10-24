LSFSCWTime = []
LSFSGetWTime = []
LSFSTime = []
LSFSObejectV = []
LSFSWObejectV = []


LSDFTime = []
PRPCTime = []
SSelectTime = []
LaplacianScoreTime = []
FisherScoreTime = []


def timeit(time_list):
    def _deco(fun):
        print("add decorator timeit to {0}".format(fun.__name__))
        _deco.__name__ = fun.__name__
        def __deco(*args, **kwargs):
            import time
            s = time.clock()
            result = fun(*args, **kwargs)
            dual = time.clock() - s
            time_list.append(dual)
            return result
        __deco.__name__ = fun.__name__
        return __deco
    return _deco


def get_object_value(obejectVList):
    def _deco(fun):
        print("add decorator get_object_value to {0}".format(fun.__name__))
        _deco.__name__ = fun.__name__
        import copy
        def __deco(*args, **kwargs):
            result = fun(*args, **kwargs)
            obejectVList.append(copy.deepcopy(result))
            return result
        __deco.__name__ = fun.__name__
        return __deco
    return _deco


def reset_lsfs_global_value(fun):
    print("add decorator reset_lsfs_global_value to {0}".format(fun.__name__))
    def _deco(*args, **kwargs):
        global LSFSGetWTime, LSFSCWTime, LSFSWObejectV, LSFSObejectV, LSFSTime, LSFSFW
        del LSFSCWTime[:]
        del LSFSGetWTime[:]
        del LSFSTime[:]
        # LSFSTime = []这样来重置是错误的。LSFSTime = []相当于给了LSFSTime一块新的内存。
        # 但如果外部引用了原有的内存，那么对外部的内存并未重置。所以要用del LSFSTime[:]来删除
        del LSFSObejectV[:]
        del LSFSWObejectV[:]
        result = fun(*args, **kwargs)
        return result
    _deco.__name__ = fun.__name__
    return _deco



def reset_lsdf_global_value(fun):
    def _deco(*args, **kwargs):
        _deco.__name__ = fun.__name__
        global LSDFTime
        del LSDFTime[:]
        result = fun(*args, **kwargs)
        return result
    return _deco


def reset_PRPC_global_value(fun):
    def _deco(*args, **kwargs):
        _deco.__name__ = fun.__name__
        global PRPCTime
        del PRPCTime[:]
        result = fun(*args, **kwargs)
        return result
    return _deco


def reset_SSelect_global_value(fun):
    def _deco(*args, **kwargs):
        _deco.__name__ = fun.__name__
        global SSelectTime
        del SSelectTime[:]
        result = fun(*args, **kwargs)
        return result
    return _deco


def reset_LaplacianScore_global_value(fun):
    def _deco(*args, **kwargs):
        _deco.__name__ = fun.__name__
        global LaplacianScoreTime
        del LaplacianScoreTime[:]
        result = fun(*args, **kwargs)
        return result
    return _deco


def reset_FisherScore_global_value(fun):
    def _deco(*args, **kwargs):
        _deco.__name__ = fun.__name__
        global FisherScoreTime
        del FisherScoreTime[:]
        result = fun(*args, **kwargs)
        return result
    return _deco


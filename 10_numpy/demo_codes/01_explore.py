
import numpy as np 
import pandas as pd 

# %%

def build_index_array(m):
    import itertools

    return np.fromiter(
        itertools.product(range(m), range(m)), 
        dtype=[("axis0", np.int32), ("axis1", np.int32)]
    ).reshape(m, m)

# %%

def check_spacing():
    pos_zero_uint32 = np.float32(0.0).view(np.uint32)
    pos_inf_uint32 = np.float32(np.inf).view(np.uint32)

    neg_zero_uint32 = np.float32(-0.0).view(np.uint32)
    neg_inf_uint32 = np.float32(-np.inf).view(np.uint32)

    group_size = 10_000_000

    for start_uint32 in range(pos_zero_uint32, pos_inf_uint32, group_size):
        end_uint32 = start_uint32 + group_size
        if end_uint32 > pos_inf_uint32:
            end_uint32 = pos_inf_uint32

        numbers = np.arange(start_uint32, end_uint32, dtype=np.uint32).view(np.float32)
        num_exceptions = np.sum(np.spacing(numbers) != np.nextafter(numbers, np.inf) - numbers)

        assert num_exceptions == 0
    
    for start_uint32 in range(neg_zero_uint32 + 1, neg_inf_uint32, group_size):
        end_uint32 = start_uint32 + group_size
        if end_uint32 > neg_inf_uint32:
            end_uint32 = neg_inf_uint32
        
        numbers = np.arange(start_uint32, end_uint32, dtype=np.uint32).view(np.float32)
        num_exceptions = np.sum(np.spacing(numbers) != np.nextafter(numbers, -np.inf) - numbers)

        assert num_exceptions == 0


# %%

def show_ufunc():

    rows = []

    for ufunc in dir(np):
        ufunc = getattr(np, ufunc)
        if not isinstance(ufunc, np.ufunc):
            continue

        rows.append([
            ufunc.__name__, ufunc.nin, ufunc.nout, ufunc.nargs, 
            ufunc.ntypes, ufunc.types, ufunc.identity, ufunc.signature
        ])
    
    df = pd.DataFrame(rows, columns=[
        "name", "nin", "nout", "nargs", 
        "ntypes", "types", "identity", "signature"
    ])

    df.to_excel("ufunc.xlsx")

    print(np.dtype(np.float32).char)

# %%

def test_ufunc():

    for ufunc in dir(np):
        ufunc = getattr(np, ufunc)
        if not isinstance(ufunc, np.ufunc):
            continue

        if ufunc.nin != 2 or ufunc.nout != 1 or ufunc.signature is not None:
            continue

        success = 0

        try:
            a = np.random.randn(10, 10)
            ufunc.reduce(a, axis=0)
        except TypeError:
            pass 
        else:
            success += 1
        
        try:
            a = np.random.randint(1, 10, (10, 10))
            ufunc.reduce(a, axis=0)
        except TypeError:
            pass 
        else:
            success += 1
        
        print(ufunc.__name__, success)

# %%

def test_unwrap():

    def unwrap(array, axis: int = -1, period: float = 2 * np.pi):

        def fix_range(x1, x2):
            # 将 x2 修正到 [x1 - period / 2, x1 + period / 2] 的范围内
            half_period = period / 2 

            if x2 < x1 - half_period:
                dis = x1 + half_period - x2  # x2 到 x1 + half_period 之间的距离
                step = dis // period - (dis % period == 0)
                return x2 + step * period
            if x2 > x1 + half_period:
                dis = x2 - (x1 - half_period)  # x1 - half_period 到 x2 之间的距离
                step = dis // period - (dis % period == 0)
                return x2 - step * period

            return x2 

        func = np.frompyfunc(fix_range, nin=2, nout=1).accumulate

        return func(array, axis=axis).astype(np.float64)

    a = np.random.randn(1000, 1000) * 5
    r1 = unwrap(a, axis=0)
    r2 = np.unwrap(a, axis=0)
    print((r1 != r2).sum(), np.abs(r1 - r2).max())

    a = np.array([
        [0, -181],
        [0, 181],
        [0, 180],
        [0, 540]
    ])
    print(unwrap(a, axis=1, period=360))
    print(np.unwrap(a, axis=1, period=360))
        

# %%

def test_unwrap():

    def unwrap(array, axis: int = -1, period: float = 2 * np.pi):

        def fix_range(x1, x2):
            # 将 x2 修正到 [x1 - period / 2, x1 + period / 2] 的范围内
            half_period = period / 2 

            dis = x2 - (x1 - half_period)  # x2 到 x1 - half_period 之间的距离
            step = dis // period
            return x2 - step * period

        func = np.frompyfunc(fix_range, nin=2, nout=1).accumulate

        return func(array, axis=axis).astype(np.float64)

    a = np.random.randn(1000, 1000) * 5
    r1 = unwrap(a, axis=0)
    r2 = np.unwrap(a, axis=0)
    print((r1 != r2).sum(), np.abs(r1 - r2).max())

    a = np.array([
        [0, -181],
        [0, 181],
        [0, 180],
        [0, -180],
        [0, 540],
        [0, -540]
    ])
    print(unwrap(a, axis=1, period=360))
    print(np.unwrap(a, axis=1, period=360))
    print(unwrap(a, axis=1, period=-360))

# %%

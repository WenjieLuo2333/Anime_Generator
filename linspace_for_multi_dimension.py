"""
Implement the Interpolation of vectors.
"""
import numpy as np
def linspace_two(v1,v2,sample_nums):
    """
    v1,v2: array(size = (1,s))  the two vectors to make interpolation
    sample_nums: int the number of interpolate points
    ans: array(size = (sample_nums,s))
    """
    ans = [[0 for _ in range(sample_nums)] for i in range(np.shape(v1)[0])]
    for i in range(len(ans)):
        ans[i][:] = np.linspace(v1[i],v2[i],sample_nums)
    return np.array(ans).T

def linspace_four(lt,rt,ld,rd,sample_nums):
    """
    lt - left top. rt - right top. ....
    lt,rt,ld,rd: array(size = (1,s))  the two vectors to make interpolation
    sample_nums: int the number of interpolate points
    ans: array(size = (sample_nums*sample_nums,s))
    interpolated maxtrix = 
    [
        lt, a01,a02, ... , rt
        a10, ...        , a1n
        ..                 ..
        ld,..           ..,rd
    ]
    each element is a vector in size (1,s)
    ans = [lt,a01,a02,...,rt,a10,...,a1n,..,..,ld,..,rd]
    """
    lcol = linspace_two(lt,ld,sample_nums)
    rcol = linspace_two(rt,rd,sample_nums)
    ans = []
    for i in range(len(lcol)):
        ans.append(linspace_two(lcol[i],rcol[i],sample_nums))
    res = [_ for i in ans for _ in i]
    return np.array(res)
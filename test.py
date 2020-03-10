import gpucommunity
import numpy as np
import numpy.testing as npt

def test():
    # Initialize arrays
    arr = np.array([1,2,2,2], dtype=np.int32)
    adder = gpucommunity.GPUCommunity(arr)
    adder.increment()
    
    # compute
    adder.retreive_inplace() # modifies adder
    got = adder.retreive()
    want = [2,3,3,3]

    #Verify
    npt.assert_array_equal(arr, want)
    npt.assert_array_equal(got, want)
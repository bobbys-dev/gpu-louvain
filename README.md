## cython wrapped CUDA/C++

This code makes an explicit cython class that wraps the C++ class, exposing it in python.

To install:

`$ python setup.py install`

Install these in a virtual environment:
`$ pip install requirements.txt` 

Test:

`$ pytest test.py`

you need a relatively recent version of cython (>=0.16) and CUDA sm_35
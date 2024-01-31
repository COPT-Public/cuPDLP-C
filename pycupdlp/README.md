## Installation
To install **Python** interface of cuPDLP-C, you should compile the target **pycupdlp**. Then run command in **pycupdlp** directory:

```bash
python setup.py install
```
## Usage
You can solve **mps** files simply like:

```python
import pycupdlp
c = pycupdlp.cupdlp()
c.readMPS('path to your mps file')
c.setParams({'ifScaling':0})
c.solve()
```

where you can set parameters by python dict using function **set_params**. And we provide helper function **helper**.

You can also formulate the LP by explicitly providing the coefficient matrix and vectors, where coefficient matrix should be **scipy** sparse matrix, and vectors can be list or **numpy** array. We provide a detailed example in **example.py**.






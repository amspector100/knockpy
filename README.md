# Knockadapt

## To run tests

- To run all tests, run ``python3 -m pytest`` 
- To run a specific label, run ``pytest -v -m {label}``.
- To select all labels except a particular one, run ``pytest -v -m "not {label}"`` (with the quotes).
- To run a specific file, try pytest test/{file_name}.py. To run a specific test within the file, run pytest test/{file_name}.py::classname::test_method. You also don't have to specify
the exact test_method, you get the idea.
- To run a test with profiling, try ``python3 -m pytest {path} --profile``. This should generate a set of .prof files in prof/. Then you can run snakeviz filename.prof to visualize the output.
There are also more flags/options for outputs in the command line command.
- Alternatively, cprofilev is much better.
To run cprofilev, copy and paste the test to proftest/* and then run 
``python3 -m cprofilev proftest/test_name.py``.


## To do

1. Graphs --> ren to dgp, class based
2. 

### Knockoff Generation

1. Gaussian knockoff generator should be class based
2. There should be an overarching "sample knockoffs"
function where you can put the type of knockoffs
you want to sample in as an input argument.
3. It would be cool if we moved the KS test code and used
it as a method to validate the knockoffs.

### FX Knockoff Support

1. Knockoff Filter + Debiased Lasso
2. Need to think about whether we'll actually shift X

### Knockoff Construction

1. Add hierarchical clustering to ASDP group-making

### MCV Computation

1. Gradient-based method can be sped up
2. Add value for rec_prop

### Graphs

1. DGP class? instead of returning like 6 things?
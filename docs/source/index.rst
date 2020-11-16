.. knockpy documentation master file, created by
   sphinx-quickstart on Sun Nov 15 15:23:29 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to knockpy's documentation!
===================================

Knockoffs are a powerful tool which can be used in combination with nearly any machine learning algorithm to control the false discovery rate (FDR), initially developed by `Barber and Candes 2015`_ and `Candes et al 2018`_.

.. _Candes et al 2018: https://arxiv.org/abs/1610.02351
.. _Barber and Candes 2015: https://projecteuclid.org/download/pdfview_1/euclid.aos/1438606853

Installation
============

To install knockpy, first install choldate:
``pip install git+git://github.com/jcrudy/choldate.git``

Then, install knockpy using pip:
``pip install knockpy``

To use the (optional) kpytorch submodule, you will need to install `pytorch`_. If the installation fails on your system, please reach out to me and I'll try to help.

.. _pytorch: https://pytorch.org/

Quickstart
==========

Given a data-matrix `X` and a response vector `y`, knockpy makes it easy to use knockoffs to perform variable selection using a wide variety of machine learning algorithms (also known as "feature statistic") and types of knockoffs. One quick example is shown below, where we use the cross-validated lasso to assign variable importances to the features and knockoffs.  

.. code-block:: python

	import knockpy as kpy
	from knockpy.knockoff_filter import KnockoffFilter

	# Generate synthetic data from a Gaussian linear model
	data_gen_process = kpy.dgp.DGP()
	data_gen_process.sample_data(
		n=1500, # Number of datapoints
		p=500, # Dimensionality
		sparsity=0.1,
		x_dist='gaussian',
	)
	X = data_gen_process.X
	y = data_gen_process.y
	Sigma=data_gen_process.Sigma

	# Run model-X knockoffs
	kfilter = KnockoffFilter(
		fstat='lasso',
		ksampler='gaussian',
	)
	rejections = kfilter.forward(X=X, y=y, Sigma=Sigma)

Most importantly, knockpy is built to be modular, so researchers and analysts can easily layer functionality on top of it. See [usage] for more details!


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   kfilter
   ksamplers
   kstats
   smatrix
   dgp
   kpytorch


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

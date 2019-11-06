#!/bin/sh
python -m unittest -v test.test_suites.ds_suite test.test_suites.snn_suite test.test_suites.pynn_brian_suite
#python -m unittest -v test.test_suites.pynn_brian_suite
#python -m unittest -v -f test.test_suites.ds_suite
#python -m unittest -v test.test_suites.snn_suite
#python -m unittest -v test.test_suites.ds_suite test.test_suites.snn_suite

#!/bin/sh
echo "Running tests using DS backend"
#python -m unittest -f test.test_suites.ds_suite
python3 -m unittest -f test.test_suites.ds_suite
echo '>>>---<<<'

echo "Running tests using SNN backend"
#python -m unittest -f test.test_suites.snn_suite
#python3 -m unittest -f test.test_suites.snn_suite
echo '>>>---<<<'

#python -m unittest -f test.test_suites.pynn_spinnaker_suite
echo "Running tests using Pynn-Brian backend"
#python2 -m unittest -f test.test_suites.pynn_brian_suite
echo '>>>---<<<'

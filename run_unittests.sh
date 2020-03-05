#!/bin/sh
#echo $1
ds_tests() {
    echo "Running tests using DS backend"
    python3 -m unittest -f test.test_suites.ds_suite
    echo '>>>---<<<'
}

snn_tests() {
    echo "Running tests using SNN backend"
    python3 -m unittest -f test.test_suites.snn_suite
    echo '>>>---<<<'
}

pynn_brian_tests() {
    echo "Running tests using Pynn-Brian backend"
    python2 -m unittest -f test.test_suites.pynn_brian_suite
    echo '>>>---<<<'
}

case $1 in 
    "ds")
        ds_tests
        ;;

    "snn")
        snn_tests
        ;;

    "pynn")
        pynn_brian_tests
        ;;

    "brian")
        pynn_brian_tests
        ;;

    *)
        ds_tests
        snn_tests
        pynn_brian_tests
esac

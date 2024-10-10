#!/bin/bash

javac @argfile mr_demo/$1.java
java @argfile mr_demo.$1
rm -rf mr_demo/*.class

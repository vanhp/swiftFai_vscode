import XCTest

import batchnormTests

var tests = [XCTestCaseEntry]()
tests += batchnormTests.allTests()
XCTMain(tests)

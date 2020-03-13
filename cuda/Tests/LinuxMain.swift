import XCTest

import cudaTests

var tests = [XCTestCaseEntry]()
tests += cudaTests.allTests()
XCTMain(tests)

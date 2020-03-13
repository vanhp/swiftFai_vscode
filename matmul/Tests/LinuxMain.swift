import XCTest

import matmulTests

var tests = [XCTestCaseEntry]()
tests += matmulTests.allTests()
XCTMain(tests)

import XCTest

import annealingTests

var tests = [XCTestCaseEntry]()
tests += annealingTests.allTests()
XCTMain(tests)

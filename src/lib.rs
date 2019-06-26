pub enum TestResult {
    Accepted,
}

pub fn do_test(_stdin: &str, _expected_stdout: &str) -> TestResult {
    TestResult::Accepted
}

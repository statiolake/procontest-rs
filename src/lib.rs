use cli_test_dir::{CommandExt, TestDir};
use std::{cmp, fmt, iter, usize};

const BINARY_NAME: &str = "./main";

const EPS: f64 = 1e-8;

#[derive(Debug, PartialEq)]
pub enum TestResult {
    Accepted,
    PresentationError,
    WrongAnswer(Box<WrongAnswer>),
    RuntimeError(RuntimeErrorKind),
}

#[derive(Debug, PartialEq)]
pub struct WrongAnswer {
    pub context: Context,
    pub details: Vec<WrongAnswerKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineColumn {
    line: usize,
    column: usize,
}

impl LineColumn {
    pub fn new(line: usize, column: usize) -> LineColumn {
        LineColumn { line, column }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span(LineColumn, LineColumn);

impl Span {
    fn start(&self) -> LineColumn {
        self.0
    }

    fn end(&self) -> LineColumn {
        self.1
    }
}

#[derive(Debug, PartialEq)]
pub enum WrongAnswerKind {
    NumOfLineDiffers {
        expected: usize,
        actual: usize,
    },
    NumOfTokenDiffers {
        expected: usize,
        expected_line_len: usize,
        actual: usize,
        actual_line_len: usize,
        lineno: usize,
    },
    TokenDiffers {
        expected: Token,
        actual: Token,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub enum RuntimeErrorKind {
    CommandFailed,
    InvalidUtf8,
}

pub fn do_test(test_id: &str, stdin: &str, expected: &str) -> TestResult {
    let test_dir = TestDir::new(BINARY_NAME, test_id);

    let output = match test_dir.cmd().output_with_stdin(stdin) {
        Ok(output) => output,
        Err(_e) => return TestResult::RuntimeError(RuntimeErrorKind::CommandFailed),
    };

    let actual = match String::from_utf8(output.stdout) {
        Ok(stdout) => stdout,
        Err(_e) => return TestResult::RuntimeError(RuntimeErrorKind::InvalidUtf8),
    };

    let stderr = match String::from_utf8(output.stderr) {
        Ok(stderr) => stderr,
        Err(_e) => return TestResult::RuntimeError(RuntimeErrorKind::InvalidUtf8),
    };

    let expected = split_into_lines(&expected).map(|x| x.to_string()).collect();
    let actual = split_into_lines(&actual).map(|x| x.to_string()).collect();

    Context::new(expected, actual, stderr).verify()
}

#[derive(Debug, PartialEq)]
enum VerifyResult {
    Pass,
    Fail(TestResult),
}

macro_rules! ensure_pass {
    ($e:expr) => {
        if let VerifyResult::Fail(e) = $e {
            return e;
        }
    };
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Context {
    pub expected: Vec<String>,
    pub actual: Vec<String>,
    pub stderr: String,

    is_presentation_error: bool,
}

impl Context {
    pub fn new(expected: Vec<String>, actual: Vec<String>, stderr: String) -> Context {
        Context {
            expected,
            actual,
            stderr,
            is_presentation_error: false,
        }
    }

    pub fn verify(mut self) -> TestResult {
        self.fix();
        ensure_pass!(self.verify_num_lines(self.expected.len(), self.actual.len()));

        let zipped = self.expected.iter().zip(self.actual.iter());
        let mut errors = Vec::new();
        for (lineno, (expected, actual)) in zipped.enumerate() {
            errors.extend(self.verify_line(expected, actual, lineno));
        }
        if !errors.is_empty() {
            return TestResult::WrongAnswer(Box::new(WrongAnswer {
                context: self.clone(),
                details: errors,
            }));
        }

        if self.is_presentation_error {
            return TestResult::PresentationError;
        }

        TestResult::Accepted
    }

    fn fix(&mut self) {
        if self.expected.ends_with(&["".into()]) {
            self.expected.pop();
        }

        if self.actual.ends_with(&["".into()]) {
            self.actual.pop();
        } else {
            self.is_presentation_error = true;
        }
    }

    fn verify_num_lines(&self, expected: usize, actual: usize) -> VerifyResult {
        if expected != actual {
            return VerifyResult::Fail(TestResult::WrongAnswer(Box::new(WrongAnswer {
                context: self.clone(),
                details: vec![WrongAnswerKind::NumOfLineDiffers { expected, actual }],
            })));
        }

        VerifyResult::Pass
    }

    fn verify_line(
        &self,
        expected_line: &str,
        actual_line: &str,
        lineno: usize,
    ) -> Vec<WrongAnswerKind> {
        let expected = Token::parse_line(expected_line, lineno);
        let actual = Token::parse_line(actual_line, lineno);

        if expected.len() != actual.len() {
            return vec![WrongAnswerKind::NumOfTokenDiffers {
                expected: expected.len(),
                actual: actual.len(),
                expected_line_len: expected_line.len(),
                actual_line_len: actual_line.len(),
                lineno,
            }];
        }

        let mut errors = vec![];
        for (expected, actual) in expected.iter().zip(actual.iter()) {
            if !self.compare_token(expected, actual) {
                errors.push(WrongAnswerKind::TokenDiffers {
                    expected: expected.clone(),
                    actual: actual.clone(),
                });
            }
        }

        errors
    }

    fn compare_token(&self, a: &Token, b: &Token) -> bool {
        match (&a.kind, &b.kind) {
            (TokenKind::String(a), TokenKind::String(b)) => a == b,
            (TokenKind::Uint(a), TokenKind::Uint(b)) => a == b,
            (TokenKind::Int(a), TokenKind::Int(b)) => a == b,
            (TokenKind::Float(a), TokenKind::Float(b)) => (a - b).abs() < EPS,
            _ => false,
        }
    }
}

fn split_into_lines(s: &str) -> impl Iterator<Item = &str> {
    let mut iter = s.split('\n');
    assert_eq!(
        iter.next_back(),
        Some(""),
        "expected or actual has no lines.  This is a bug of the *procontest*."
    );
    iter
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    kind: TokenKind,
    span: Span,
}

impl Token {
    fn new(kind: TokenKind, span: Span) -> Token {
        Token { kind, span }
    }
}

impl fmt::Display for Token {
    fn fmt(&self, b: &mut fmt::Formatter) -> fmt::Result {
        match &self.kind {
            TokenKind::String(v) => write!(b, "{}", v),
            TokenKind::Uint(v) => write!(b, "{}", v),
            TokenKind::Int(v) => write!(b, "{}", v),
            TokenKind::Float(v) => write!(b, "{}", v),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    String(String),
    Uint(u64),
    Int(i64),
    Float(f64),
}

impl Token {
    fn parse_line(line: &str, lineno: usize) -> Vec<Token> {
        // if it contains two successive whitespace or starts with or ends with whitespace, treat
        // entire line as string literal.
        if line.contains("  ") || line.starts_with(' ') || line.ends_with(' ') {
            return vec![Token::new(
                TokenKind::String(line.into()),
                Span(
                    LineColumn::new(lineno, 0),
                    LineColumn::new(lineno, line.len()),
                ),
            )];
        }

        // otherwise, split tokens and parse each token
        let mut iter = line.char_indices().peekable();

        iter::from_fn(|| match iter.peek() {
            None => None,
            Some(_) => Some(
                iter.by_ref()
                    .skip_while(|(_, ch)| ch.is_whitespace())
                    .take_while(|(_, ch)| !ch.is_whitespace())
                    .fold(
                        (usize::MAX, usize::MIN, "".to_string()),
                        |(start, end, mut current), (columnno, ch)| {
                            current.push(ch);
                            (
                                cmp::min(start, columnno),
                                cmp::max(end, columnno + 1),
                                current,
                            )
                        },
                    ),
            ),
        })
        .map(|(startno, endno, token)| {
            Token::parse(
                &token,
                LineColumn::new(lineno, startno),
                LineColumn::new(lineno, endno),
            )
        })
        .collect()
    }

    fn parse(token: &str, start: LineColumn, end: LineColumn) -> Token {
        // A token starting with zero is rarely intended to be a number so treat it as a stirng.
        // If it were not handled specially, the value would be parsed as an integer.
        if token != "0"
            && token != "-0"
            && !token.starts_with("0.")
            && !token.starts_with("-0.")
            && (token.starts_with('0') || token.starts_with("-0"))
        {
            return Token::new(TokenKind::String(token.into()), Span(start, end));
        }

        if let Ok(uint) = token.parse() {
            return Token::new(TokenKind::Uint(uint), Span(start, end));
        }

        if let Ok(int) = token.parse() {
            return Token::new(TokenKind::Int(int), Span(start, end));
        }

        if let Ok(float) = token.parse() {
            return Token::new(TokenKind::Float(float), Span(start, end));
        }

        Token::new(TokenKind::String(token.into()), Span(start, end))
    }
}

pub fn format(result: TestResult) -> String {
    match result {
        TestResult::Accepted => "Accepted.".into(),
        TestResult::PresentationError => "Presentation error.".into(),
        TestResult::WrongAnswer(wa) => format!(
            "Wrong Answer.\n\nexpected stdout:\n\n{}\nactual stdout:\n\n{}\nerrors:\n\n{}",
            joinl(&wa.context.expected),
            joinl(&wa.context.actual),
            format_wa(*wa)
        ),
        TestResult::RuntimeError(re) => format!("Runtime Error: {}", format_re(re)),
    }
}

fn format_wa(wa: WrongAnswer) -> String {
    let mut messages = Vec::new();
    let mut expected_spans = Vec::new();
    let mut actual_spans = Vec::new();
    for detail in wa.details {
        messages.push(match detail {
            WrongAnswerKind::NumOfLineDiffers { expected, actual } => format!(
                "The number of lines is different. expected: {}, actual: {}",
                expected, actual
            ),

            WrongAnswerKind::NumOfTokenDiffers {
                expected,
                expected_line_len,
                actual,
                actual_line_len,
                lineno,
            } => {
                expected_spans.push(Span(
                    LineColumn {
                        line: lineno,
                        column: 0,
                    },
                    LineColumn {
                        line: lineno,
                        column: expected_line_len,
                    },
                ));
                actual_spans.push(Span(
                    LineColumn {
                        line: lineno,
                        column: 0,
                    },
                    LineColumn {
                        line: lineno,
                        column: actual_line_len,
                    },
                ));
                format!(
                    "At line {}: the number of tokens is different. expected: {}, actual: {}",
                    lineno + 1,
                    expected,
                    actual
                )
            }

            WrongAnswerKind::TokenDiffers { expected, actual } => {
                expected_spans.push(expected.span);
                actual_spans.push(actual.span);
                format!(
                    "At line {}: Token differs. expected: {}, actual: {}",
                    expected.span.start().line + 1,
                    expected,
                    actual,
                )
            }
        })
    }

    let messages = joinl(&messages);
    let diff = format_diff(
        wa.context.expected,
        wa.context.actual,
        expected_spans,
        actual_spans,
    );

    format!("{}\n{}", messages, diff)
}

fn joinl(ss: &[String]) -> String {
    let mut res = String::new();
    for s in ss {
        res.push_str(s);
        res.push('\n');
    }
    res
}

fn format_diff(
    expected: Vec<String>,
    actual: Vec<String>,
    expected_spans: Vec<Span>,
    actual_spans: Vec<Span>,
) -> String {
    fn fallback(_expected: Vec<String>, _actual: Vec<String>) -> String {
        "WARNING: Cannot determine a terminal size or too narrow terminal.  Cannot use diff view."
            .into()
    }

    let lineno_delim = " | ";
    let max_lineno = cmp::max(expected.len(), actual.len());
    let lineno_width = max_lineno.to_string().len();
    let division_delim = " | ";
    let decoration_width = lineno_width + lineno_delim.len() + division_delim.len();

    use terminal_size::*;
    let width = match terminal_size() {
        Some((Width(w), _)) => w as usize,
        None => return fallback(expected, actual),
    };

    if decoration_width >= width {
        return fallback(expected, actual);
    }
    let half = (width - decoration_width) / 2;

    use splitv::Pane;
    let expected_len: Vec<_> = expected.iter().map(|x| x.len()).collect();
    let actual_len: Vec<_> = actual.iter().map(|x| x.len()).collect();

    let expected_len_max = expected_len.iter().max().copied().unwrap_or(0);
    let actual_len_max = actual_len.iter().max().copied().unwrap_or(0);

    let body = {
        let lineno_pane = Pane {
            lines: (1..=max_lineno).map(|x| x.to_string()).collect(),
            width: lineno_width,
        };

        let expected_pane = Pane {
            lines: expected,
            width: cmp::min(half, expected_len_max),
        };
        let actual_pane = Pane {
            lines: actual,
            width: cmp::min(half, actual_len_max),
        };

        splitv::splitv(
            vec![lineno_pane, expected_pane, actual_pane],
            vec![lineno_delim.to_string(), division_delim.to_string()],
        )
    };

    let span = {
        let organize_spans = |spans: Vec<Span>| -> Vec<Vec<(usize, usize)>> {
            let mut organized = vec![Vec::new(); max_lineno];
            for span in spans {
                assert_eq!(span.start().line, span.end().line);
                organized[span.start().line].push((span.start().column, span.end().column));
            }
            organized
        };

        let convert_span = |line_width: usize, spans: Vec<(usize, usize)>| -> String {
            let mut line = " ".repeat(line_width);
            for (start, end) in spans {
                line.replace_range(start..end, &"^".repeat(end - start));
            }
            line
        };

        let lineno_pane = Pane {
            lines: vec![" ".to_string(); max_lineno],
            width: lineno_width,
        };

        let expected_pane = Pane {
            lines: organize_spans(expected_spans)
                .into_iter()
                .enumerate()
                .map(|(i, spans)| convert_span(expected_len.get(i).copied().unwrap_or(0), spans))
                .collect(),
            width: cmp::min(half, expected_len_max),
        };

        let actual_pane = Pane {
            lines: organize_spans(actual_spans)
                .into_iter()
                .enumerate()
                .map(|(i, spans)| convert_span(actual_len.get(i).copied().unwrap_or(0), spans))
                .collect(),
            width: cmp::min(half, actual_len_max),
        };

        splitv::splitv(
            vec![lineno_pane, expected_pane, actual_pane],
            vec![lineno_delim.to_string(), division_delim.to_string()],
        )
    };

    use itertools::Itertools;
    joinl(
        &body
            .into_iter()
            .interleave(span.into_iter())
            .collect::<Vec<_>>(),
    )
}

fn format_re(re: RuntimeErrorKind) -> String {
    match re {
        RuntimeErrorKind::CommandFailed => "The process did not exit successfully.".into(),
        RuntimeErrorKind::InvalidUtf8 => "Process outputs invalid UTF-8.".into(),
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[allow(non_snake_case)]
    fn S(s: &str) -> String {
        String::from(s)
    }

    #[test]
    fn accepted_simple() {
        assert_eq!(
            TestResult::Accepted,
            Context::new(vec![S("1"), S("")], vec![S("1"), S("")], S("")).verify()
        );

        assert_eq!(
            TestResult::Accepted,
            Context::new(vec![S("1 "), S("")], vec![S("1 "), S("")], S("")).verify()
        );

        assert_eq!(
            TestResult::Accepted,
            Context::new(vec![S("1 2 3"), S("")], vec![S("1 2 3"), S("")], S("")).verify()
        );

        assert_eq!(
            TestResult::Accepted,
            Context::new(vec![S("011121"), S("")], vec![S("011121"), S("")], S("")).verify()
        );

        assert_eq!(
            TestResult::Accepted,
            Context::new(
                vec![S("12312131231231235534254351121"), S("")],
                vec![S("12312131231231235534254351121"), S("")],
                S("")
            )
            .verify()
        );

        assert_eq!(
            TestResult::Accepted,
            Context::new(
                vec![S("1"), S("2 3"), S("4 5"), S("")],
                vec![S("1"), S("2 3"), S("4 5"), S("")],
                S("")
            )
            .verify()
        );
    }

    #[test]
    fn accepted_float() {
        assert_eq!(
            TestResult::Accepted,
            Context::new(
                vec![S("0.123123123123123"), S("")],
                vec![S("0.123123123123129"), S("")],
                S("")
            )
            .verify()
        );
    }

    #[test]
    fn presentation_error() {
        assert_eq! {
            TestResult::PresentationError,
            Context::new(vec![], vec![], S("")).verify()
        }
    }

    fn fixed(ctx: &Context) -> Context {
        let mut ctx = ctx.clone();
        ctx.fix();
        ctx
    }

    #[test]
    fn wrong_answer_num_of_line_differs() {
        let ctx = Context::new(vec![S("1"), S("2"), S("")], vec![S("2"), S("")], S(""));
        assert_eq!(
            TestResult::WrongAnswer(Box::new(WrongAnswer {
                context: fixed(&ctx),
                details: vec![WrongAnswerKind::NumOfLineDiffers {
                    expected: 2,
                    actual: 1,
                }]
            })),
            ctx.verify(),
        );
    }

    #[test]
    fn wrong_answer_num_of_token_differs() {
        let ctx = Context::new(vec![S("1 2"), S("")], vec![S("2"), S("")], S(""));
        assert_eq!(
            TestResult::WrongAnswer(Box::new(WrongAnswer {
                context: fixed(&ctx),
                details: vec![WrongAnswerKind::NumOfTokenDiffers {
                    expected: 2,
                    actual: 1,
                    expected_line_len: 3,
                    actual_line_len: 1,
                    lineno: 0,
                }]
            })),
            ctx.verify(),
        );

        let ctx = Context::new(vec![S("1 2"), S("")], vec![S("2 3 4"), S("")], S(""));
        assert_eq!(
            TestResult::WrongAnswer(Box::new(WrongAnswer {
                context: fixed(&ctx),
                details: vec![WrongAnswerKind::NumOfTokenDiffers {
                    expected: 2,
                    actual: 3,
                    expected_line_len: 3,
                    actual_line_len: 5,
                    lineno: 0,
                }]
            })),
            ctx.verify(),
        );

        let ctx = Context::new(vec![S("1 2"), S("")], vec![S("1  2"), S("")], S(""));
        assert_eq!(
            TestResult::WrongAnswer(Box::new(WrongAnswer {
                context: fixed(&ctx),
                details: vec![WrongAnswerKind::NumOfTokenDiffers {
                    expected: 2,
                    actual: 1,
                    expected_line_len: 3,
                    actual_line_len: 4,
                    lineno: 0,
                }]
            })),
            ctx.verify(),
        );
    }

    #[test]
    fn wrong_answer_token_differs() {
        let ctx = Context::new(vec![S("1"), S("")], vec![S("2"), S("")], S(""));
        assert_eq!(
            TestResult::WrongAnswer(Box::new(WrongAnswer {
                context: fixed(&ctx),
                details: vec![WrongAnswerKind::TokenDiffers {
                    expected: Token::new(
                        TokenKind::Uint(1),
                        Span(LineColumn::new(0, 0), LineColumn::new(0, 1))
                    ),
                    actual: Token::new(
                        TokenKind::Uint(2),
                        Span(LineColumn::new(0, 0), LineColumn::new(0, 1))
                    )
                }]
            })),
            ctx.verify(),
        );

        let ctx = Context::new(vec![S("00"), S("")], vec![S("0"), S("")], S(""));
        assert_eq!(
            TestResult::WrongAnswer(Box::new(WrongAnswer {
                context: fixed(&ctx),
                details: vec![WrongAnswerKind::TokenDiffers {
                    expected: Token::new(
                        TokenKind::String(S("00")),
                        Span(LineColumn::new(0, 0), LineColumn::new(0, 2))
                    ),
                    actual: Token::new(
                        TokenKind::Uint(0),
                        Span(LineColumn::new(0, 0), LineColumn::new(0, 1))
                    )
                }]
            })),
            ctx.verify(),
        );

        let ctx = Context::new(vec![S("1.0003"), S("")], vec![S("1.0002"), S("")], S(""));
        assert_eq!(
            TestResult::WrongAnswer(Box::new(WrongAnswer {
                context: fixed(&ctx),
                details: vec![WrongAnswerKind::TokenDiffers {
                    expected: Token::new(
                        TokenKind::Float(1.0003),
                        Span(LineColumn::new(0, 0), LineColumn::new(0, 6))
                    ),
                    actual: Token::new(
                        TokenKind::Float(1.0002),
                        Span(LineColumn::new(0, 0), LineColumn::new(0, 6))
                    )
                }]
            })),
            ctx.verify(),
        );

        let ctx = Context::new(vec![S("-0.0003"), S("")], vec![S("-0.0002"), S("")], S(""));
        assert_eq!(
            TestResult::WrongAnswer(Box::new(WrongAnswer {
                context: fixed(&ctx),
                details: vec![WrongAnswerKind::TokenDiffers {
                    expected: Token::new(
                        TokenKind::Float(-0.0003),
                        Span(LineColumn::new(0, 0), LineColumn::new(0, 7))
                    ),
                    actual: Token::new(
                        TokenKind::Float(-0.0002),
                        Span(LineColumn::new(0, 0), LineColumn::new(0, 7))
                    )
                }]
            })),
            ctx.verify(),
        );

        let ctx = Context::new(vec![S("-1"), S("")], vec![S("-01"), S("")], S(""));
        assert_eq!(
            TestResult::WrongAnswer(Box::new(WrongAnswer {
                context: fixed(&ctx),
                details: vec![WrongAnswerKind::TokenDiffers {
                    expected: Token::new(
                        TokenKind::Int(-1),
                        Span(LineColumn::new(0, 0), LineColumn::new(0, 2))
                    ),
                    actual: Token::new(
                        TokenKind::String(S("-01")),
                        Span(LineColumn::new(0, 0), LineColumn::new(0, 3))
                    )
                }]
            })),
            ctx.verify(),
        );

        let ctx = Context::new(vec![S("3 2 -1"), S("")], vec![S("3 2 -01"), S("")], S(""));
        assert_eq!(
            TestResult::WrongAnswer(Box::new(WrongAnswer {
                context: fixed(&ctx),
                details: vec![WrongAnswerKind::TokenDiffers {
                    expected: Token::new(
                        TokenKind::Int(-1),
                        Span(LineColumn::new(0, 4), LineColumn::new(0, 6))
                    ),
                    actual: Token::new(
                        TokenKind::String(S("-01")),
                        Span(LineColumn::new(0, 4), LineColumn::new(0, 7))
                    )
                }]
            })),
            ctx.verify(),
        );

        let ctx = Context::new(vec![S("1 2 -1"), S("")], vec![S("4 3 -01"), S("")], S(""));
        assert_eq!(
            TestResult::WrongAnswer(Box::new(WrongAnswer {
                context: fixed(&ctx),
                details: vec![
                    WrongAnswerKind::TokenDiffers {
                        expected: Token::new(
                            TokenKind::Uint(1),
                            Span(LineColumn::new(0, 0), LineColumn::new(0, 1))
                        ),
                        actual: Token::new(
                            TokenKind::Uint(4),
                            Span(LineColumn::new(0, 0), LineColumn::new(0, 1))
                        )
                    },
                    WrongAnswerKind::TokenDiffers {
                        expected: Token::new(
                            TokenKind::Uint(2),
                            Span(LineColumn::new(0, 2), LineColumn::new(0, 3))
                        ),
                        actual: Token::new(
                            TokenKind::Uint(3),
                            Span(LineColumn::new(0, 2), LineColumn::new(0, 3))
                        )
                    },
                    WrongAnswerKind::TokenDiffers {
                        expected: Token::new(
                            TokenKind::Int(-1),
                            Span(LineColumn::new(0, 4), LineColumn::new(0, 6))
                        ),
                        actual: Token::new(
                            TokenKind::String(S("-01")),
                            Span(LineColumn::new(0, 4), LineColumn::new(0, 7))
                        )
                    }
                ]
            })),
            ctx.verify(),
        );
    }

    #[test]
    fn formatting() {
        let ctx = Context::new(
            vec![S("1 2 -1"), S("2   3"), S("asdf jkl fsd"), S("")],
            vec![S("4 3 -01"), S("2 3"), S("asdf jkl fsh"), S("")],
            S(""),
        );
        println!("{}", format(ctx.verify()));
    }
}

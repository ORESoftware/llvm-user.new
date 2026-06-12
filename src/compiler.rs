//! The Quandary programming language compiler.
//!
//! Pipeline:  source.qq  ->  tokens  ->  AST  ->  LLVM IR (text)  ->  clang  ->  native binary
//!
//! Language features:
//!   * first-class functions (functions are values)
//!   * lambda functions  `fn(x: Int) -> Int { return x * 2; }`
//!   * closures          lambdas capture surrounding variables by value (heap env)
//!   * no implicit return — every function returns only via an explicit `return`
//!   * 100% memory safe   — there is no `null` in the language. Absence is modelled with
//!                          the `Option[T]` type, and the only way to read the inner value
//!                          is to `match` on it, so an NPE is unrepresentable.
//!
//! Everything after the parser is handed to LLVM (register allocation, instruction
//! selection, optimization, x86/ARM backends, linking).

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::process::Command;

// ----------------------------------------------------------------------------
// Errors
// ----------------------------------------------------------------------------

fn err(msg: impl std::fmt::Display) -> ! {
    eprintln!("quandary: error: {msg}");
    std::process::exit(1);
}

// ----------------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Type {
    Int,
    Bool,
    Opt(Box<Type>),
    /// function type: parameter types + return type (the closure environment is implicit)
    Func(Vec<Type>, Box<Type>),
}

impl Type {
    /// LLVM type used to *store / pass* a value of this Quandary type.
    fn llvm(&self) -> String {
        match self {
            Type::Int => "i64".to_string(),
            Type::Bool => "i1".to_string(),
            Type::Opt(t) => format!("{{ i1, {} }}", t.llvm()),
            // a first-class function value is a closure pair: { code ptr, env ptr }
            Type::Func(_, _) => "{ ptr, ptr }".to_string(),
        }
    }
}

// ----------------------------------------------------------------------------
// Lexer
// ----------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Int(i64),
    Ident(String),
    // keywords
    KFn,      // fn
    KLet,     // let
    KReturn,  // return
    KIf,      // if
    KElse,    // else
    KWhile,   // while
    KTrue,    // true
    KFalse,   // false
    KMatch,   // match
    KSome,    // some
    KNone,    // none
    KInt,     // Int
    KBool,    // Bool
    KOption,  // Option
    KFnType,  // Fn
    // punctuation / operators
    LParen, RParen, LBrace, RBrace, LBrack, RBrack,
    Comma, Semi, Colon, Arrow, FatArrow,
    Assign, Eq, Ne, Lt, Le, Gt, Ge,
    Plus, Minus, Star, Slash, Not, And, Or,
    Eof,
}

fn lex(src: &str) -> Vec<Tok> {
    let b = src.as_bytes();
    let mut i = 0;
    let n = b.len();
    let mut toks = Vec::new();
    while i < n {
        let c = b[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        // line comments
        if c == '/' && i + 1 < n && b[i + 1] == b'/' {
            while i < n && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // numbers
        if c.is_ascii_digit() {
            let start = i;
            while i < n && (b[i] as char).is_ascii_digit() {
                i += 1;
            }
            let s = &src[start..i];
            let v: i64 = s.parse().unwrap_or_else(|_| err(format!("bad integer literal `{s}`")));
            toks.push(Tok::Int(v));
            continue;
        }
        // identifiers / keywords
        if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < n && ((b[i] as char).is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let s = &src[start..i];
            let t = match s {
                "fn" => Tok::KFn,
                "let" => Tok::KLet,
                "return" => Tok::KReturn,
                "if" => Tok::KIf,
                "else" => Tok::KElse,
                "while" => Tok::KWhile,
                "true" => Tok::KTrue,
                "false" => Tok::KFalse,
                "match" => Tok::KMatch,
                "some" => Tok::KSome,
                "none" => Tok::KNone,
                "Int" => Tok::KInt,
                "Bool" => Tok::KBool,
                "Option" => Tok::KOption,
                "Fn" => Tok::KFnType,
                _ => Tok::Ident(s.to_string()),
            };
            toks.push(t);
            continue;
        }
        // operators / punctuation (two-char first)
        let two = if i + 1 < n { &src[i..i + 2] } else { "" };
        let t = match two {
            "->" => Some(Tok::Arrow),
            "=>" => Some(Tok::FatArrow),
            "==" => Some(Tok::Eq),
            "!=" => Some(Tok::Ne),
            "<=" => Some(Tok::Le),
            ">=" => Some(Tok::Ge),
            "&&" => Some(Tok::And),
            "||" => Some(Tok::Or),
            _ => None,
        };
        if let Some(t) = t {
            toks.push(t);
            i += 2;
            continue;
        }
        let t = match c {
            '(' => Tok::LParen,
            ')' => Tok::RParen,
            '{' => Tok::LBrace,
            '}' => Tok::RBrace,
            '[' => Tok::LBrack,
            ']' => Tok::RBrack,
            ',' => Tok::Comma,
            ';' => Tok::Semi,
            ':' => Tok::Colon,
            '=' => Tok::Assign,
            '<' => Tok::Lt,
            '>' => Tok::Gt,
            '+' => Tok::Plus,
            '-' => Tok::Minus,
            '*' => Tok::Star,
            '/' => Tok::Slash,
            '!' => Tok::Not,
            other => err(format!("unexpected character `{other}`")),
        };
        toks.push(t);
        i += 1;
    }
    toks.push(Tok::Eof);
    toks
}

// ----------------------------------------------------------------------------
// AST
// ----------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Expr {
    Int(i64),
    Bool(bool),
    Ident(String),
    Unary(String, Box<Expr>),
    Binary(String, Box<Expr>, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Lambda(Vec<(String, Type)>, Type, Vec<Stmt>),
    Some_(Box<Expr>),
    None_(Type),
    /// match scrut { some(x) => a, none => b }
    Match(Box<Expr>, String, Box<Expr>, Box<Expr>),
}

#[derive(Clone, Debug)]
enum Stmt {
    Let(String, Option<Type>, Expr),
    Return(Expr),
    Assign(String, Expr),
    ExprStmt(Expr),
    If(Expr, Vec<Stmt>, Option<Vec<Stmt>>),
    While(Expr, Vec<Stmt>),
}

#[derive(Clone, Debug)]
struct FnDecl {
    name: String,
    params: Vec<(String, Type)>,
    ret: Type,
    body: Vec<Stmt>,
}

// ----------------------------------------------------------------------------
// Parser (recursive descent + Pratt for expressions)
// ----------------------------------------------------------------------------

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
}

impl Parser {
    fn new(toks: Vec<Tok>) -> Self {
        Parser { toks, pos: 0 }
    }
    fn peek(&self) -> &Tok {
        &self.toks[self.pos]
    }
    fn next(&mut self) -> Tok {
        let t = self.toks[self.pos].clone();
        self.pos += 1;
        t
    }
    fn eat(&mut self, t: Tok) {
        if *self.peek() == t {
            self.pos += 1;
        } else {
            err(format!("expected {:?}, found {:?}", t, self.peek()));
        }
    }

    fn parse_program(&mut self) -> Vec<FnDecl> {
        let mut fns = Vec::new();
        while *self.peek() != Tok::Eof {
            self.eat(Tok::KFn);
            let name = self.ident();
            let params = self.parse_params();
            self.eat(Tok::Arrow);
            let ret = self.parse_type();
            let body = self.parse_block();
            fns.push(FnDecl { name, params, ret, body });
        }
        fns
    }

    fn ident(&mut self) -> String {
        match self.next() {
            Tok::Ident(s) => s,
            other => err(format!("expected identifier, found {other:?}")),
        }
    }

    fn parse_params(&mut self) -> Vec<(String, Type)> {
        self.eat(Tok::LParen);
        let mut params = Vec::new();
        while *self.peek() != Tok::RParen {
            let name = self.ident();
            self.eat(Tok::Colon);
            let ty = self.parse_type();
            params.push((name, ty));
            if *self.peek() == Tok::Comma {
                self.next();
            } else {
                break;
            }
        }
        self.eat(Tok::RParen);
        params
    }

    fn parse_type(&mut self) -> Type {
        match self.next() {
            Tok::KInt => Type::Int,
            Tok::KBool => Type::Bool,
            Tok::KOption => {
                self.eat(Tok::LBrack);
                let inner = self.parse_type();
                self.eat(Tok::RBrack);
                Type::Opt(Box::new(inner))
            }
            Tok::KFnType => {
                self.eat(Tok::LParen);
                let mut ps = Vec::new();
                while *self.peek() != Tok::RParen {
                    ps.push(self.parse_type());
                    if *self.peek() == Tok::Comma {
                        self.next();
                    } else {
                        break;
                    }
                }
                self.eat(Tok::RParen);
                self.eat(Tok::Arrow);
                let ret = self.parse_type();
                Type::Func(ps, Box::new(ret))
            }
            other => err(format!("expected type, found {other:?}")),
        }
    }

    fn parse_block(&mut self) -> Vec<Stmt> {
        self.eat(Tok::LBrace);
        let mut stmts = Vec::new();
        while *self.peek() != Tok::RBrace {
            stmts.push(self.parse_stmt());
        }
        self.eat(Tok::RBrace);
        stmts
    }

    fn parse_stmt(&mut self) -> Stmt {
        match self.peek() {
            Tok::KLet => {
                self.next();
                let name = self.ident();
                let ann = if *self.peek() == Tok::Colon {
                    self.next();
                    Some(self.parse_type())
                } else {
                    None
                };
                self.eat(Tok::Assign);
                let e = self.parse_expr(0);
                self.eat(Tok::Semi);
                Stmt::Let(name, ann, e)
            }
            Tok::KReturn => {
                self.next();
                let e = self.parse_expr(0);
                self.eat(Tok::Semi);
                Stmt::Return(e)
            }
            Tok::KIf => {
                self.next();
                self.eat(Tok::LParen);
                let cond = self.parse_expr(0);
                self.eat(Tok::RParen);
                let then = self.parse_block();
                let els = if *self.peek() == Tok::KElse {
                    self.next();
                    Some(self.parse_block())
                } else {
                    None
                };
                Stmt::If(cond, then, els)
            }
            Tok::KWhile => {
                self.next();
                self.eat(Tok::LParen);
                let cond = self.parse_expr(0);
                self.eat(Tok::RParen);
                let body = self.parse_block();
                Stmt::While(cond, body)
            }
            // assignment `name = expr;`  vs  expression statement
            Tok::Ident(_) if self.toks.get(self.pos + 1) == Some(&Tok::Assign) => {
                let name = self.ident();
                self.eat(Tok::Assign);
                let e = self.parse_expr(0);
                self.eat(Tok::Semi);
                Stmt::Assign(name, e)
            }
            _ => {
                let e = self.parse_expr(0);
                self.eat(Tok::Semi);
                Stmt::ExprStmt(e)
            }
        }
    }

    // Pratt parser ----------------------------------------------------------
    fn infix_bp(t: &Tok) -> Option<(u8, &'static str)> {
        Some(match t {
            Tok::Or => (1, "||"),
            Tok::And => (2, "&&"),
            Tok::Eq => (3, "=="),
            Tok::Ne => (3, "!="),
            Tok::Lt => (4, "<"),
            Tok::Le => (4, "<="),
            Tok::Gt => (4, ">"),
            Tok::Ge => (4, ">="),
            Tok::Plus => (5, "+"),
            Tok::Minus => (5, "-"),
            Tok::Star => (6, "*"),
            Tok::Slash => (6, "/"),
            _ => return None,
        })
    }

    fn parse_expr(&mut self, min_bp: u8) -> Expr {
        let mut lhs = self.parse_prefix();
        loop {
            // postfix call
            if *self.peek() == Tok::LParen {
                let args = self.parse_args();
                lhs = Expr::Call(Box::new(lhs), args);
                continue;
            }
            let Some((bp, op)) = Self::infix_bp(self.peek()) else { break };
            if bp < min_bp {
                break;
            }
            self.next();
            let rhs = self.parse_expr(bp + 1); // left-associative
            lhs = Expr::Binary(op.to_string(), Box::new(lhs), Box::new(rhs));
        }
        lhs
    }

    fn parse_args(&mut self) -> Vec<Expr> {
        self.eat(Tok::LParen);
        let mut args = Vec::new();
        while *self.peek() != Tok::RParen {
            args.push(self.parse_expr(0));
            if *self.peek() == Tok::Comma {
                self.next();
            } else {
                break;
            }
        }
        self.eat(Tok::RParen);
        args
    }

    fn parse_prefix(&mut self) -> Expr {
        match self.peek() {
            Tok::Minus => {
                self.next();
                Expr::Unary("-".into(), Box::new(self.parse_prefix()))
            }
            Tok::Not => {
                self.next();
                Expr::Unary("!".into(), Box::new(self.parse_prefix()))
            }
            _ => self.parse_primary(),
        }
    }

    fn parse_primary(&mut self) -> Expr {
        match self.next() {
            Tok::Int(v) => Expr::Int(v),
            Tok::KTrue => Expr::Bool(true),
            Tok::KFalse => Expr::Bool(false),
            Tok::Ident(s) => Expr::Ident(s),
            Tok::LParen => {
                let e = self.parse_expr(0);
                self.eat(Tok::RParen);
                e
            }
            // lambda:  fn(params) -> Ret { body }
            Tok::KFn => {
                let params = self.parse_params();
                self.eat(Tok::Arrow);
                let ret = self.parse_type();
                let body = self.parse_block();
                Expr::Lambda(params, ret, body)
            }
            Tok::KSome => {
                self.eat(Tok::LParen);
                let e = self.parse_expr(0);
                self.eat(Tok::RParen);
                Expr::Some_(Box::new(e))
            }
            Tok::KNone => {
                self.eat(Tok::LBrack);
                let t = self.parse_type();
                self.eat(Tok::RBrack);
                Expr::None_(t)
            }
            Tok::KMatch => {
                let scrut = self.parse_expr(0);
                self.eat(Tok::LBrace);
                // some(x) => expr
                self.eat(Tok::KSome);
                self.eat(Tok::LParen);
                let bind = self.ident();
                self.eat(Tok::RParen);
                self.eat(Tok::FatArrow);
                let some_arm = self.parse_expr(0);
                self.eat(Tok::Comma);
                // none => expr
                self.eat(Tok::KNone);
                self.eat(Tok::FatArrow);
                let none_arm = self.parse_expr(0);
                // optional trailing comma
                if *self.peek() == Tok::Comma {
                    self.next();
                }
                self.eat(Tok::RBrace);
                Expr::Match(
                    Box::new(scrut),
                    bind,
                    Box::new(some_arm),
                    Box::new(none_arm),
                )
            }
            other => err(format!("unexpected token in expression: {other:?}")),
        }
    }
}

// ----------------------------------------------------------------------------
// Free-variable analysis (for closure capture)
// ----------------------------------------------------------------------------

fn free_expr(e: &Expr, bound: &HashSet<String>, free: &mut HashSet<String>) {
    match e {
        Expr::Int(_) | Expr::Bool(_) | Expr::None_(_) => {}
        Expr::Ident(n) => {
            if !bound.contains(n) {
                free.insert(n.clone());
            }
        }
        Expr::Unary(_, a) => free_expr(a, bound, free),
        Expr::Binary(_, a, b) => {
            free_expr(a, bound, free);
            free_expr(b, bound, free);
        }
        Expr::Call(c, args) => {
            free_expr(c, bound, free);
            for a in args {
                free_expr(a, bound, free);
            }
        }
        Expr::Some_(a) => free_expr(a, bound, free),
        Expr::Lambda(params, _, body) => {
            let mut b2 = bound.clone();
            for (p, _) in params {
                b2.insert(p.clone());
            }
            free_block(body, &b2, free);
        }
        Expr::Match(scrut, x, sa, na) => {
            free_expr(scrut, bound, free);
            let mut b2 = bound.clone();
            b2.insert(x.clone());
            free_expr(sa, &b2, free);
            free_expr(na, bound, free);
        }
    }
}

fn free_block(stmts: &[Stmt], bound: &HashSet<String>, free: &mut HashSet<String>) {
    let mut b = bound.clone();
    for s in stmts {
        match s {
            Stmt::Let(n, _, init) => {
                free_expr(init, &b, free);
                b.insert(n.clone());
            }
            Stmt::Return(e) | Stmt::ExprStmt(e) => free_expr(e, &b, free),
            Stmt::Assign(n, e) => {
                free_expr(e, &b, free);
                if !b.contains(n) {
                    free.insert(n.clone());
                }
            }
            Stmt::If(c, t, el) => {
                free_expr(c, &b, free);
                free_block(t, &b, free);
                if let Some(el) = el {
                    free_block(el, &b, free);
                }
            }
            Stmt::While(c, body) => {
                free_expr(c, &b, free);
                free_block(body, &b, free);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Code generator (AST -> LLVM IR text)
// ----------------------------------------------------------------------------

#[derive(Clone)]
struct Var {
    slot: String, // llvm pointer holding the value (an alloca)
    ty: Type,
}

/// per-function code generation state
struct Ctx {
    body: String,
    tmp: usize,
    lbl: usize,
    cur: String,        // current basic block label (no leading %)
    terminated: bool,   // current block already has a terminator
    ret: Type,
    scope: Vec<HashMap<String, Var>>,
}

impl Ctx {
    fn new(ret: Type) -> Self {
        Ctx {
            body: String::new(),
            tmp: 0,
            lbl: 0,
            cur: "entry".to_string(),
            terminated: false,
            ret,
            scope: vec![HashMap::new()],
        }
    }
    fn tmp(&mut self) -> String {
        let s = format!("%t{}", self.tmp);
        self.tmp += 1;
        s
    }
    fn label(&mut self) -> String {
        let s = format!("L{}", self.lbl);
        self.lbl += 1;
        s
    }
    fn emit(&mut self, s: impl AsRef<str>) {
        if self.terminated {
            return; // drop dead instructions after a terminator
        }
        self.body.push_str("  ");
        self.body.push_str(s.as_ref());
        self.body.push('\n');
    }
    fn place_label(&mut self, name: &str) {
        let _ = writeln!(self.body, "{name}:");
        self.cur = name.to_string();
        self.terminated = false;
    }
    fn br(&mut self, target: &str) {
        if self.terminated {
            return;
        }
        let _ = writeln!(self.body, "  br label %{target}");
        self.terminated = true;
    }
    fn br_cond(&mut self, c: &str, a: &str, b: &str) {
        if self.terminated {
            return;
        }
        let _ = writeln!(self.body, "  br i1 {c}, label %{a}, label %{b}");
        self.terminated = true;
    }
    fn push_scope(&mut self) {
        self.scope.push(HashMap::new());
    }
    fn pop_scope(&mut self) {
        self.scope.pop();
    }
    fn declare(&mut self, name: &str, slot: String, ty: Type) {
        self.scope
            .last_mut()
            .unwrap()
            .insert(name.to_string(), Var { slot, ty });
    }
    fn lookup(&self, name: &str) -> Option<Var> {
        for s in self.scope.iter().rev() {
            if let Some(v) = s.get(name) {
                return Some(v.clone());
            }
        }
        None
    }
}

struct Compiler {
    /// global function signatures: name -> (param types, return type)
    globals: HashMap<String, (Vec<Type>, Type)>,
    /// finished function definitions
    fns: Vec<String>,
    /// module-level helper type defs (closure environments)
    type_defs: String,
    lambda_id: usize,
}

impl Compiler {
    fn new() -> Self {
        Compiler {
            globals: HashMap::new(),
            fns: Vec::new(),
            type_defs: String::new(),
            lambda_id: 0,
        }
    }

    fn compile_program(&mut self, prog: &[FnDecl]) -> String {
        for f in prog {
            let params = f.params.iter().map(|(_, t)| t.clone()).collect();
            self.globals.insert(f.name.clone(), (params, f.ret.clone()));
        }
        if !self.globals.contains_key("main") {
            err("program has no `fn main() -> Int`");
        }

        for f in prog {
            let text = self.compile_function(
                &format!("q_{}", f.name),
                &[],
                &f.params,
                &f.ret,
                &f.body,
            );
            self.fns.push(text);
        }

        self.assemble()
    }

    /// Compile one function (top-level or lifted lambda) to an LLVM definition.
    /// `captures` are loaded from the implicit env pointer at entry.
    fn compile_function(
        &mut self,
        sym: &str,                  // llvm symbol without leading @
        captures: &[(String, Type)],
        params: &[(String, Type)],
        ret: &Type,
        body: &[Stmt],
    ) -> String {
        let mut ctx = Ctx::new(ret.clone());

        // signature: every function takes a leading env pointer (ignored if no captures)
        let mut sig = String::from("ptr %env");
        for (i, (_, ty)) in params.iter().enumerate() {
            let _ = write!(sig, ", {} %arg{i}", ty.llvm());
        }

        // entry: materialize captures from the env, then params, into allocas
        if !captures.is_empty() {
            let env_ty = format!("%env.{sym}");
            for (i, (name, ty)) in captures.iter().enumerate() {
                let p = ctx.tmp();
                ctx.emit(format!(
                    "{p} = getelementptr {env_ty}, ptr %env, i32 0, i32 {i}"
                ));
                let v = ctx.tmp();
                ctx.emit(format!("{v} = load {ty}, ptr {p}", ty = ty.llvm()));
                let slot = ctx.tmp();
                ctx.emit(format!("{slot} = alloca {}", ty.llvm()));
                ctx.emit(format!("store {} {v}, ptr {slot}", ty.llvm()));
                ctx.declare(name, slot, ty.clone());
            }
        }
        for (i, (name, ty)) in params.iter().enumerate() {
            let slot = ctx.tmp();
            ctx.emit(format!("{slot} = alloca {}", ty.llvm()));
            ctx.emit(format!("store {} %arg{i}, ptr {slot}", ty.llvm()));
            ctx.declare(name, slot, ty.clone());
        }

        self.gen_stmts(&mut ctx, body);

        // no implicit return: a fall-through means the programmer forgot a return.
        // Make it a deterministic abort rather than undefined behavior.
        if !ctx.terminated {
            ctx.emit("call void @abort()");
            ctx.emit("unreachable");
            ctx.terminated = true;
        }

        format!(
            "define {ret} @{sym}({sig}) {{\nentry:\n{body}}}\n",
            ret = ret.llvm(),
            body = ctx.body
        )
    }

    fn gen_stmts(&mut self, ctx: &mut Ctx, stmts: &[Stmt]) {
        for s in stmts {
            if ctx.terminated {
                break;
            }
            self.gen_stmt(ctx, s);
        }
    }

    fn gen_block(&mut self, ctx: &mut Ctx, stmts: &[Stmt]) {
        ctx.push_scope();
        self.gen_stmts(ctx, stmts);
        ctx.pop_scope();
    }

    fn gen_stmt(&mut self, ctx: &mut Ctx, s: &Stmt) {
        match s {
            Stmt::Let(name, ann, init) => {
                let (v, ty) = self.gen_expr(ctx, init);
                let ty = ann.clone().unwrap_or(ty);
                let slot = ctx.tmp();
                ctx.emit(format!("{slot} = alloca {}", ty.llvm()));
                ctx.emit(format!("store {} {v}, ptr {slot}", ty.llvm()));
                ctx.declare(name, slot, ty);
            }
            Stmt::Assign(name, e) => {
                let var = ctx
                    .lookup(name)
                    .unwrap_or_else(|| err(format!("assignment to unknown variable `{name}`")));
                let (v, _) = self.gen_expr(ctx, e);
                ctx.emit(format!("store {} {v}, ptr {}", var.ty.llvm(), var.slot));
            }
            Stmt::Return(e) => {
                let (v, _) = self.gen_expr(ctx, e);
                ctx.emit(format!("ret {} {v}", ctx.ret.llvm()));
                ctx.terminated = true;
            }
            Stmt::ExprStmt(e) => {
                self.gen_expr(ctx, e);
            }
            Stmt::If(cond, then, els) => {
                let (c, _) = self.gen_expr(ctx, cond);
                let then_l = ctx.label();
                let end_l = ctx.label();
                let else_l = if els.is_some() { ctx.label() } else { end_l.clone() };
                ctx.br_cond(&c, &then_l, &else_l);

                ctx.place_label(&then_l);
                self.gen_block(ctx, then);
                let then_term = ctx.terminated;
                ctx.br(&end_l);

                let mut else_term = false;
                if let Some(els) = els {
                    ctx.place_label(&else_l);
                    self.gen_block(ctx, els);
                    else_term = ctx.terminated;
                    ctx.br(&end_l);
                }

                ctx.place_label(&end_l);
                // if both arms returned, `end` is unreachable
                let reachable = !then_term || (els.is_some() && !else_term) || els.is_none();
                if !reachable {
                    ctx.emit("unreachable");
                    ctx.terminated = true;
                }
            }
            Stmt::While(cond, body) => {
                let cond_l = ctx.label();
                let body_l = ctx.label();
                let end_l = ctx.label();
                ctx.br(&cond_l);
                ctx.place_label(&cond_l);
                let (c, _) = self.gen_expr(ctx, cond);
                ctx.br_cond(&c, &body_l, &end_l);
                ctx.place_label(&body_l);
                self.gen_block(ctx, body);
                ctx.br(&cond_l);
                ctx.place_label(&end_l);
            }
        }
    }

    /// Returns (llvm operand, type).
    fn gen_expr(&mut self, ctx: &mut Ctx, e: &Expr) -> (String, Type) {
        match e {
            Expr::Int(v) => (v.to_string(), Type::Int),
            Expr::Bool(b) => ((if *b { "true" } else { "false" }).to_string(), Type::Bool),
            Expr::Ident(name) => {
                if let Some(var) = ctx.lookup(name) {
                    let r = ctx.tmp();
                    ctx.emit(format!("{r} = load {}, ptr {}", var.ty.llvm(), var.slot));
                    (r, var.ty)
                } else if let Some((ps, ret)) = self.globals.get(name).cloned() {
                    // first-class reference to a top-level function: build closure {fn, null}
                    let fty = Type::Func(ps, Box::new(ret));
                    let c0 = ctx.tmp();
                    ctx.emit(format!(
                        "{c0} = insertvalue {{ ptr, ptr }} undef, ptr @q_{name}, 0"
                    ));
                    let c1 = ctx.tmp();
                    ctx.emit(format!("{c1} = insertvalue {{ ptr, ptr }} {c0}, ptr null, 1"));
                    (c1, fty)
                } else {
                    err(format!("unknown identifier `{name}`"));
                }
            }
            Expr::Unary(op, a) => {
                let (v, ty) = self.gen_expr(ctx, a);
                let r = ctx.tmp();
                match op.as_str() {
                    "-" => ctx.emit(format!("{r} = sub i64 0, {v}")),
                    "!" => ctx.emit(format!("{r} = xor i1 {v}, true")),
                    _ => unreachable!(),
                }
                (r, ty)
            }
            Expr::Binary(op, a, b) => self.gen_binary(ctx, op, a, b),
            Expr::Some_(inner) => {
                let (v, t) = self.gen_expr(ctx, inner);
                let opt = Type::Opt(Box::new(t.clone()));
                let s0 = ctx.tmp();
                ctx.emit(format!(
                    "{s0} = insertvalue {ot} undef, i1 true, 0",
                    ot = opt.llvm()
                ));
                let s1 = ctx.tmp();
                ctx.emit(format!(
                    "{s1} = insertvalue {ot} {s0}, {it} {v}, 1",
                    ot = opt.llvm(),
                    it = t.llvm()
                ));
                (s1, opt)
            }
            Expr::None_(inner) => {
                let opt = Type::Opt(Box::new(inner.clone()));
                let s0 = ctx.tmp();
                // tag = false; payload left as zeroinitializer (never read without a match)
                ctx.emit(format!(
                    "{s0} = insertvalue {ot} zeroinitializer, i1 false, 0",
                    ot = opt.llvm()
                ));
                (s0, opt)
            }
            Expr::Match(scrut, bind, some_arm, none_arm) => {
                self.gen_match(ctx, scrut, bind, some_arm, none_arm)
            }
            Expr::Lambda(params, ret, body) => self.gen_lambda(ctx, params, ret, body),
            Expr::Call(callee, args) => self.gen_call(ctx, callee, args),
        }
    }

    fn gen_binary(&mut self, ctx: &mut Ctx, op: &str, a: &Expr, b: &Expr) -> (String, Type) {
        let (lv, lt) = self.gen_expr(ctx, a);
        let (rv, _rt) = self.gen_expr(ctx, b);
        let r = ctx.tmp();
        let (instr, ty) = match op {
            "+" => (format!("add i64 {lv}, {rv}"), Type::Int),
            "-" => (format!("sub i64 {lv}, {rv}"), Type::Int),
            "*" => (format!("mul i64 {lv}, {rv}"), Type::Int),
            "/" => (format!("sdiv i64 {lv}, {rv}"), Type::Int),
            "==" => (format!("icmp eq {} {lv}, {rv}", lt.llvm()), Type::Bool),
            "!=" => (format!("icmp ne {} {lv}, {rv}", lt.llvm()), Type::Bool),
            "<" => (format!("icmp slt i64 {lv}, {rv}"), Type::Bool),
            "<=" => (format!("icmp sle i64 {lv}, {rv}"), Type::Bool),
            ">" => (format!("icmp sgt i64 {lv}, {rv}"), Type::Bool),
            ">=" => (format!("icmp sge i64 {lv}, {rv}"), Type::Bool),
            "&&" => (format!("and i1 {lv}, {rv}"), Type::Bool),
            "||" => (format!("or i1 {lv}, {rv}"), Type::Bool),
            _ => unreachable!("bad binop {op}"),
        };
        ctx.emit(format!("{r} = {instr}"));
        (r, ty)
    }

    fn gen_match(
        &mut self,
        ctx: &mut Ctx,
        scrut: &Expr,
        bind: &str,
        some_arm: &Expr,
        none_arm: &Expr,
    ) -> (String, Type) {
        let (ov, ot) = self.gen_expr(ctx, scrut);
        let inner = match &ot {
            Type::Opt(t) => (**t).clone(),
            _ => err("`match` scrutinee must be an Option"),
        };
        let opt_ll = ot.llvm();
        let tag = ctx.tmp();
        ctx.emit(format!("{tag} = extractvalue {opt_ll} {ov}, 0"));
        let val = ctx.tmp();
        ctx.emit(format!("{val} = extractvalue {opt_ll} {ov}, 1"));

        let some_l = ctx.label();
        let none_l = ctx.label();
        let end_l = ctx.label();
        ctx.br_cond(&tag, &some_l, &none_l);

        // some(x) => ...
        ctx.place_label(&some_l);
        ctx.push_scope();
        let slot = ctx.tmp();
        ctx.emit(format!("{slot} = alloca {}", inner.llvm()));
        ctx.emit(format!("store {} {val}, ptr {slot}", inner.llvm()));
        ctx.declare(bind, slot, inner.clone());
        let (sv, rty) = self.gen_expr(ctx, some_arm);
        ctx.pop_scope();
        let some_pred = ctx.cur.clone();
        ctx.br(&end_l);

        // none => ...
        ctx.place_label(&none_l);
        let (nv, _) = self.gen_expr(ctx, none_arm);
        let none_pred = ctx.cur.clone();
        ctx.br(&end_l);

        // merge
        ctx.place_label(&end_l);
        let r = ctx.tmp();
        ctx.emit(format!(
            "{r} = phi {rt} [ {sv}, %{some_pred} ], [ {nv}, %{none_pred} ]",
            rt = rty.llvm()
        ));
        (r, rty)
    }

    fn gen_lambda(
        &mut self,
        ctx: &mut Ctx,
        params: &[(String, Type)],
        ret: &Type,
        body: &[Stmt],
    ) -> (String, Type) {
        // 1. determine captured variables (free vars resolved in the enclosing scope)
        let mut bound: HashSet<String> = params.iter().map(|(p, _)| p.clone()).collect();
        let mut free = HashSet::new();
        free_block(body, &bound, &mut free);
        bound.clear();
        let mut names: Vec<String> = free
            .into_iter()
            .filter(|n| !self.globals.contains_key(n) && n != "print")
            .collect();
        names.sort();
        let captures: Vec<(String, Type)> = names
            .into_iter()
            .map(|n| {
                let v = ctx
                    .lookup(&n)
                    .unwrap_or_else(|| err(format!("closure captures unknown variable `{n}`")));
                (n, v.ty)
            })
            .collect();

        let id = self.lambda_id;
        self.lambda_id += 1;
        let sym = format!("q_lambda_{id}");
        let fty = Type::Func(
            params.iter().map(|(_, t)| t.clone()).collect(),
            Box::new(ret.clone()),
        );

        // 2. build the closure value at the creation site
        let env_operand = if captures.is_empty() {
            "null".to_string()
        } else {
            // declare the environment struct type
            let env_ty = format!("%env.{sym}");
            let field_tys: Vec<String> = captures.iter().map(|(_, t)| t.llvm()).collect();
            let _ = writeln!(
                self.type_defs,
                "{env_ty} = type {{ {} }}",
                field_tys.join(", ")
            );
            // malloc(sizeof env) using the gep-on-null sizeof trick
            let env = ctx.tmp();
            ctx.emit(format!(
                "{env} = call ptr @malloc(i64 ptrtoint (ptr getelementptr ({env_ty}, ptr null, i64 1) to i64))"
            ));
            for (i, (name, ty)) in captures.iter().enumerate() {
                let var = ctx.lookup(name).unwrap();
                let loaded = ctx.tmp();
                ctx.emit(format!("{loaded} = load {}, ptr {}", ty.llvm(), var.slot));
                let p = ctx.tmp();
                ctx.emit(format!(
                    "{p} = getelementptr {env_ty}, ptr {env}, i32 0, i32 {i}"
                ));
                ctx.emit(format!("store {} {loaded}, ptr {p}", ty.llvm()));
            }
            env
        };

        let c0 = ctx.tmp();
        ctx.emit(format!(
            "{c0} = insertvalue {{ ptr, ptr }} undef, ptr @{sym}, 0"
        ));
        let c1 = ctx.tmp();
        ctx.emit(format!(
            "{c1} = insertvalue {{ ptr, ptr }} {c0}, ptr {env_operand}, 1"
        ));

        // 3. compile the lifted lambda body (uses %env.<sym> for its captures)
        let text = self.compile_function(&sym, &captures, params, ret, body);
        self.fns.push(text);

        (c1, fty)
    }

    fn gen_call(&mut self, ctx: &mut Ctx, callee: &Expr, args: &[Expr]) -> (String, Type) {
        // builtin: print(Int) -> prints and returns its argument
        if let Expr::Ident(n) = callee {
            if n == "print" {
                let (v, t) = self.gen_expr(ctx, &args[0]);
                let r = ctx.tmp();
                ctx.emit(format!(
                    "{r} = call i32 (ptr, ...) @printf(ptr @.fmt, i64 {v})"
                ));
                let _ = r;
                return (v, t);
            }
        }

        // evaluate the args
        let arg_vals: Vec<(String, Type)> =
            args.iter().map(|a| self.gen_expr(ctx, a)).collect();

        // direct call to a known top-level function
        if let Expr::Ident(n) = callee {
            if let Some((_ps, ret)) = self.globals.get(n).cloned() {
                if ctx.lookup(n).is_none() {
                    let mut call_args = String::from("ptr null");
                    for (v, t) in &arg_vals {
                        let _ = write!(call_args, ", {} {v}", t.llvm());
                    }
                    let r = ctx.tmp();
                    ctx.emit(format!(
                        "{r} = call {ret} @q_{n}({call_args})",
                        ret = ret.llvm()
                    ));
                    return (r, ret);
                }
            }
        }

        // indirect call through a closure value
        let (cv, cty) = self.gen_expr(ctx, callee);
        let ret = match &cty {
            Type::Func(_, r) => (**r).clone(),
            _ => err("attempted to call a non-function value"),
        };
        let fp = ctx.tmp();
        ctx.emit(format!("{fp} = extractvalue {{ ptr, ptr }} {cv}, 0"));
        let env = ctx.tmp();
        ctx.emit(format!("{env} = extractvalue {{ ptr, ptr }} {cv}, 1"));

        let mut sig_tys = String::from("ptr");
        let mut call_args = format!("ptr {env}");
        for (v, t) in &arg_vals {
            let _ = write!(sig_tys, ", {}", t.llvm());
            let _ = write!(call_args, ", {} {v}", t.llvm());
        }
        let r = ctx.tmp();
        ctx.emit(format!(
            "{r} = call {ret} ({sig_tys}) {fp}({call_args})",
            ret = ret.llvm()
        ));
        (r, ret)
    }

    fn assemble(&self) -> String {
        let mut m = String::new();
        m.push_str("; Quandary -> LLVM IR\n\n");
        m.push_str(&self.type_defs);
        if !self.type_defs.is_empty() {
            m.push('\n');
        }
        m.push_str("@.fmt = private unnamed_addr constant [6 x i8] c\"%lld\\0A\\00\"\n\n");
        m.push_str("declare i32 @printf(ptr, ...)\n");
        m.push_str("declare ptr @malloc(i64)\n");
        m.push_str("declare void @abort()\n\n");
        for f in &self.fns {
            m.push_str(f);
            m.push('\n');
        }
        // real entry point: call into Quandary main, return its i64 truncated to i32
        m.push_str(
            "define i32 @main() {\nentry:\n  \
             %r = call i64 @q_main(ptr null)\n  \
             %t = trunc i64 %r to i32\n  \
             ret i32 %t\n}\n",
        );
        m
    }
}

// ----------------------------------------------------------------------------
// Driver
// ----------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut input: Option<String> = None;
    let mut out: Option<String> = None;
    let mut emit_llvm = false;
    let mut run = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--emit-llvm" => emit_llvm = true,
            "--run" => run = true,
            "-o" => {
                i += 1;
                out = Some(args.get(i).cloned().unwrap_or_else(|| err("-o needs a path")));
            }
            "-h" | "--help" => {
                println!(
                    "quandary <file.qq> [-o OUT] [--emit-llvm] [--run]\n\
                     \n  -o OUT        output binary path (default: input without extension)\
                     \n  --emit-llvm   print generated LLVM IR to stdout and stop\
                     \n  --run         run the compiled program after building"
                );
                return;
            }
            other => {
                if other.starts_with('-') {
                    err(format!("unknown flag `{other}`"));
                }
                input = Some(other.to_string());
            }
        }
        i += 1;
    }

    let path = input.unwrap_or_else(|| err("no input file (try: quandary test/main.qq --run)"));
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| err(format!("cannot read `{path}`: {e}")));

    let toks = lex(&src);
    let prog = Parser::new(toks).parse_program();
    let mut c = Compiler::new();
    let ir = c.compile_program(&prog);

    if emit_llvm {
        print!("{ir}");
        return;
    }

    // write the IR next to a temp .ll and compile with clang
    let stem = std::path::Path::new(&path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "a".to_string());
    let ll_path = std::env::temp_dir().join(format!("{stem}.ll"));
    std::fs::write(&ll_path, &ir).unwrap_or_else(|e| err(format!("cannot write IR: {e}")));

    let out_path = out.unwrap_or_else(|| {
        std::path::Path::new(&path)
            .with_extension("")
            .to_string_lossy()
            .to_string()
    });

    let status = Command::new("clang")
        .arg(&ll_path)
        .arg("-o")
        .arg(&out_path)
        .arg("-Wno-override-module")
        .status()
        .unwrap_or_else(|e| err(format!("failed to invoke clang: {e}")));
    if !status.success() {
        err("clang failed to compile the generated IR");
    }
    eprintln!("quandary: built {out_path}");

    if run {
        let st = Command::new(&out_path)
            .status()
            .unwrap_or_else(|e| err(format!("failed to run binary: {e}")));
        std::process::exit(st.code().unwrap_or(1));
    }
}

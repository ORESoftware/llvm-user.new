# Quandary

A small statically-typed language that compiles to native code through LLVM.

```
source.qq  ->  AST  ->  LLVM IR (text)  ->  clang/LLVM  ->  native executable
```

The compiler (Rust, in [`src/compiler.rs`](src/compiler.rs)) does the lexing, parsing,
closure analysis and LLVM IR emission. Everything after that — register allocation,
instruction selection, optimization, x86/ARM backends, linking — is handled by LLVM.

## Build & run

```sh
cargo build --release
./target/release/quandary test/main.qq --run      # compile and execute
./target/release/quandary test/main.qq --emit-llvm # print the generated LLVM IR
./target/release/quandary test/main.qq -o prog     # just build the binary
```

Requires `clang` (any LLVM-based clang) on `PATH`.

## Language

* **First-class functions** — functions are values; bind them, pass them, return them.
* **Lambdas** — `fn(x: Int) -> Int { return x * 2; }`.
* **Closures** — lambdas capture surrounding variables by value into a heap environment.
* **No implicit return** — a value leaves a function only through an explicit `return`;
  falling off the end aborts deterministically instead of returning garbage.
* **Memory safe, no NPEs** — there is no `null`. Absence is modelled with `Option[T]`,
  and the only way to read the inner value is to `match`, so a null dereference is
  unrepresentable.

### Types

| Quandary          | Meaning                  | LLVM            |
| ----------------- | ------------------------ | --------------- |
| `Int`             | 64-bit signed integer    | `i64`           |
| `Bool`            | boolean                  | `i1`            |
| `Option[T]`       | a `T` or nothing         | `{ i1, T }`     |
| `Fn(A, B) -> C`   | first-class function     | `{ ptr, ptr }`  (code ptr + env ptr) |

### Example

```qq
fn make_adder(n: Int) -> Fn(Int) -> Int {
  return fn(x: Int) -> Int { return x + n; };   // closure over n
}

fn safe_div(a: Int, b: Int) -> Option[Int] {
  if (b == 0) { return none[Int]; }
  return some(a / b);
}

fn main() -> Int {
  let add10 = make_adder(10);
  print(add10(5));                 // 15

  let r = safe_div(20, 4);
  let v = match r {                // unwrap is only possible via match
    some(x) => x,
    none => -1
  };
  print(v);                        // 5
  return 0;
}
```

### Syntax reference

* Declarations: `fn name(p: T, ...) -> T { ... }`
* Statements: `let x = e;`, `let x: T = e;`, `x = e;`, `return e;`,
  `if (c) { } else { }`, `while (c) { }`, expression statements.
* Expressions: integer/`true`/`false` literals, `+ - * /`, `== != < <= > >=`,
  `&& ||`, unary `-`/`!`, calls `f(a, b)`, lambdas, `some(e)`, `none[T]`,
  `match e { some(x) => e, none => e }`.
* Builtin: `print(Int)` — prints an integer and returns it.

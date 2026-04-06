# Error Handling

This chapter documents error types and handling patterns in trueno-viz.

## Error Type

```rust
use trueno_viz::error::{Error, Result};

// Result type alias
pub type Result<T> = std::result::Result<T, Error>;

// Error variants
pub enum Error {
    /// Data length mismatch (e.g., x and y have different lengths)
    DataLengthMismatch { x_len: usize, y_len: usize },

    /// Empty data provided
    EmptyData,

    /// Invalid parameter value
    InvalidParameter { name: &'static str, message: String },

    /// IO error (file operations)
    Io(std::io::Error),

    /// PNG encoding error
    PngEncode(String),

    /// Invalid color specification
    InvalidColor(String),

    /// Scale domain error (e.g., min >= max)
    InvalidDomain { min: f32, max: f32 },
}
```

## Handling Errors

### Using Result

```rust
use trueno_viz::prelude::*;
use trueno_viz::plots::ScatterPlot;

fn create_plot() -> Result<()> {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0];  // Length mismatch!

    let plot = ScatterPlot::new()
        .x(&x)
        .y(&y)
        .build()?;  // Returns Err(DataLengthMismatch)

    plot.render_to_file("output.png")?;

    Ok(())
}

fn main() {
    match create_plot() {
        Ok(()) => println!("Success!"),
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

### Using unwrap (Development)

```rust
// For quick prototyping only
let plot = ScatterPlot::new()
    .x(&x)
    .y(&y)
    .build()
    .unwrap();
```

### Using expect

```rust
// With context message
let plot = ScatterPlot::new()
    .x(&x)
    .y(&y)
    .build()
    .expect("Failed to create scatter plot");
```

## Error Display

```rust
use trueno_viz::error::Error;

let err = Error::DataLengthMismatch { x_len: 5, y_len: 3 };
println!("{}", err);
// Output: "Data length mismatch: x has 5 elements, y has 3 elements"

let err = Error::InvalidParameter {
    name: "bins",
    message: "must be positive".to_string(),
};
println!("{}", err);
// Output: "Invalid parameter 'bins': must be positive"
```

## Custom Error Handling

```rust
use trueno_viz::error::{Error, Result};

fn validate_data(x: &[f32], y: &[f32]) -> Result<()> {
    if x.is_empty() || y.is_empty() {
        return Err(Error::EmptyData);
    }
    if x.len() != y.len() {
        return Err(Error::DataLengthMismatch {
            x_len: x.len(),
            y_len: y.len(),
        });
    }
    Ok(())
}
```

## Complete API

```rust
impl Error {
    pub fn is_io_error(&self) -> bool;
    pub fn is_data_error(&self) -> bool;
}

impl std::fmt::Display for Error { ... }
impl std::error::Error for Error { ... }

impl From<std::io::Error> for Error { ... }
```

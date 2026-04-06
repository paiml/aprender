//! Wine Quality Predictor - AWS Lambda Binary
//!
//! Deployable Lambda function for wine quality prediction.
//!
//! ## Build
//!
//! ```bash
//! cargo build --release --bin wine_lambda --target aarch64-unknown-linux-musl
//! ```
//!
//! ## Deploy
//!
//! ```bash
//! cp target/aarch64-unknown-linux-musl/release/wine_lambda bootstrap
//! zip wine_lambda.zip bootstrap
//! aws lambda create-function --function-name wine-quality \
//!   --runtime provided.al2023 --architecture arm64 \
//!   --handler bootstrap --zip-file fileb://wine_lambda.zip \
//!   --role arn:aws:iam::ACCOUNT:role/lambda-role
//! ```

/// Wine physicochemical features
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WineFeatures {
    pub fixed_acidity: f32,
    pub volatile_acidity: f32,
    pub citric_acid: f32,
    pub residual_sugar: f32,
    pub chlorides: f32,
    pub free_sulfur_dioxide: f32,
    pub total_sulfur_dioxide: f32,
    pub density: f32,
    pub ph: f32,
    pub sulphates: f32,
    pub alcohol: f32,
}

impl WineFeatures {
    /// Normalize features to [0, 1] range
    fn normalize(&self) -> Vec<f32> {
        vec![
            (self.fixed_acidity - 4.0) / 12.0,
            (self.volatile_acidity - 0.1) / 1.5,
            self.citric_acid,
            (self.residual_sugar - 0.9) / 14.1,
            (self.chlorides - 0.01) / 0.59,
            (self.free_sulfur_dioxide - 1.0) / 71.0,
            (self.total_sulfur_dioxide - 6.0) / 283.0,
            (self.density - 0.99) / 0.05,
            (self.ph - 2.7) / 1.3,
            (self.sulphates - 0.3) / 1.7,
            (self.alcohol - 8.0) / 7.0,
        ]
    }
}

/// Lambda request (direct invocation)
#[derive(Debug, serde::Deserialize)]
pub struct Request {
    #[serde(flatten)]
    pub wine: WineFeatures,
}

/// Function URL event wrapper
#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FunctionUrlEvent {
    pub body: Option<String>,
    #[serde(default)]
    pub is_base64_encoded: bool,
}

/// Lambda response
#[derive(Debug, serde::Serialize)]
pub struct Response {
    pub quality: f32,
    pub category: String,
    pub confidence: f32,
    pub top_factors: Vec<String>,
}

/// Wine quality predictor (linear model)
struct WinePredictor {
    weights: Vec<f32>,
    bias: f32,
}

impl WinePredictor {
    fn new() -> Self {
        Self {
            weights: vec![
                -0.05, // fixed_acidity
                -0.85, // volatile_acidity (negative - vinegar)
                0.45,  // citric_acid (positive - freshness)
                0.02,  // residual_sugar
                -0.15, // chlorides
                0.08,  // free_sulfur_dioxide
                -0.12, // total_sulfur_dioxide
                -0.30, // density
                -0.10, // pH
                0.55,  // sulphates (positive)
                0.95,  // alcohol (strong positive)
            ],
            bias: 5.5,
        }
    }

    fn predict(&self, features: &WineFeatures) -> Response {
        let normalized = features.normalize();

        // Linear prediction
        let mut score = self.bias;
        for (w, x) in self.weights.iter().zip(normalized.iter()) {
            score += w * x;
        }
        let quality = score.clamp(0.0, 10.0);

        // Category
        let category = match quality {
            q if q < 5.0 => "Poor",
            q if q < 7.0 => "Average",
            q if q < 8.0 => "Good",
            _ => "Excellent",
        }
        .to_string();

        // Confidence
        let confidence = normalized
            .iter()
            .filter(|&&x| (0.0..=1.0).contains(&x))
            .count() as f32
            / normalized.len() as f32;

        // Top factors
        let feature_names = [
            "fixed_acidity",
            "volatile_acidity",
            "citric_acid",
            "residual_sugar",
            "chlorides",
            "free_sulfur_dioxide",
            "total_sulfur_dioxide",
            "density",
            "ph",
            "sulphates",
            "alcohol",
        ];

        let mut importance: Vec<_> = feature_names
            .iter()
            .zip(self.weights.iter().zip(normalized.iter()))
            .map(|(name, (w, x))| (*name, (w * x).abs()))
            .collect();
        importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_factors = importance
            .iter()
            .take(3)
            .map(|(name, _)| name.to_string())
            .collect();

        Response {
            quality,
            category,
            confidence,
            top_factors,
        }
    }
}

// Static predictor (cold start optimization)
static PREDICTOR: std::sync::OnceLock<WinePredictor> = std::sync::OnceLock::new();

fn get_predictor() -> &'static WinePredictor {
    PREDICTOR.get_or_init(WinePredictor::new)
}

/// Parse request from either direct invoke or Function URL event
fn parse_and_predict(raw_body: &str, predictor: &WinePredictor) -> String {
    // Try Function URL event format first
    if let Ok(event) = serde_json::from_str::<FunctionUrlEvent>(raw_body) {
        if let Some(body) = event.body {
            // Decode body if base64 encoded
            let decoded = if event.is_base64_encoded {
                let bytes = base64_decode(&body).unwrap_or_default();
                String::from_utf8(bytes).unwrap_or_default()
            } else {
                body
            };

            return match serde_json::from_str::<Request>(&decoded) {
                Ok(req) => {
                    let response = predictor.predict(&req.wine);
                    serde_json::to_string(&response)
                        .unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.to_string())
                },
                Err(e) => format!(r#"{{"error":"Invalid request body: {e}"}}"#),
            };
        }
    }

    // Try direct request format
    match serde_json::from_str::<Request>(raw_body) {
        Ok(req) => {
            let response = predictor.predict(&req.wine);
            serde_json::to_string(&response)
                .unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.to_string())
        },
        Err(e) => format!(r#"{{"error":"Invalid request: {e}"}}"#),
    }
}

/// Simple base64 decoder
fn base64_decode(input: &str) -> Option<Vec<u8>> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut output = Vec::new();
    let mut buffer = 0u32;
    let mut bits = 0;

    for byte in input.bytes() {
        if byte == b'=' {
            break;
        }
        let value = ALPHABET.iter().position(|&c| c == byte)? as u32;
        buffer = (buffer << 6) | value;
        bits += 6;

        if bits >= 8 {
            bits -= 8;
            output.push((buffer >> bits) as u8);
            buffer &= (1 << bits) - 1;
        }
    }

    Some(output)
}

fn main() {
    // Check if running in Lambda or locally
    if std::env::var("AWS_LAMBDA_RUNTIME_API").is_ok() {
        // Lambda runtime
        lambda_runtime();
    } else {
        // Local demo
        local_demo();
    }
}

fn local_demo() {
    println!("Wine Quality Predictor (Local Demo)\n");

    let predictor = get_predictor();

    // Test wines
    let wines = vec![
        (
            "Premium Bordeaux",
            WineFeatures {
                fixed_acidity: 7.4,
                volatile_acidity: 0.28,
                citric_acid: 0.45,
                residual_sugar: 2.1,
                chlorides: 0.076,
                free_sulfur_dioxide: 15.0,
                total_sulfur_dioxide: 46.0,
                density: 0.9958,
                ph: 3.35,
                sulphates: 0.68,
                alcohol: 12.8,
            },
        ),
        (
            "Budget Table Wine",
            WineFeatures {
                fixed_acidity: 8.5,
                volatile_acidity: 0.72,
                citric_acid: 0.12,
                residual_sugar: 3.8,
                chlorides: 0.092,
                free_sulfur_dioxide: 8.0,
                total_sulfur_dioxide: 28.0,
                density: 0.9972,
                ph: 3.42,
                sulphates: 0.48,
                alcohol: 10.2,
            },
        ),
    ];

    for (name, features) in wines {
        let response = predictor.predict(&features);
        println!("{name}:");
        println!(
            "  Quality: {:.2}/10 ({})",
            response.quality, response.category
        );
        println!("  Confidence: {:.0}%", response.confidence * 100.0);
        println!("  Top factors: {}\n", response.top_factors.join(", "));
    }

    // JSON example
    println!("Example JSON request:");
    let example = WineFeatures {
        fixed_acidity: 7.0,
        volatile_acidity: 0.3,
        citric_acid: 0.4,
        residual_sugar: 2.0,
        chlorides: 0.08,
        free_sulfur_dioxide: 15.0,
        total_sulfur_dioxide: 40.0,
        density: 0.995,
        ph: 3.3,
        sulphates: 0.6,
        alcohol: 12.0,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&example).unwrap_or_default()
    );
}

fn lambda_runtime() {
    // Simple Lambda runtime loop
    let runtime_api =
        std::env::var("AWS_LAMBDA_RUNTIME_API").expect("AWS_LAMBDA_RUNTIME_API not set");

    let client = ureq::agent();
    let predictor = get_predictor();

    loop {
        // Get next invocation
        let next_url = format!("http://{runtime_api}/2018-06-01/runtime/invocation/next");
        let resp = match client.get(&next_url).call() {
            Ok(r) => r,
            Err(e) => {
                eprintln!("Failed to get invocation: {e}");
                continue;
            },
        };

        let request_id = resp
            .header("Lambda-Runtime-Aws-Request-Id")
            .unwrap_or("unknown")
            .to_string();

        // Parse request - handle both direct invoke and Function URL events
        let body = resp.into_string().unwrap_or_default();
        let result = parse_and_predict(&body, predictor);

        // Send response
        let response_url =
            format!("http://{runtime_api}/2018-06-01/runtime/invocation/{request_id}/response");
        if let Err(e) = client
            .post(&response_url)
            .set("Content-Type", "application/json")
            .send_string(&result)
        {
            eprintln!("Failed to send response: {e}");
        }
    }
}

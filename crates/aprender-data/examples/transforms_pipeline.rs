#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::uninlined_format_args,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::float_cmp,
    clippy::needless_late_init,
    clippy::redundant_clone,
    clippy::doc_markdown,
    clippy::unnecessary_debug_formatting
)]
//! Transform Pipeline Example
//!
//! Demonstrates data transformation capabilities:
//! - Column selection and dropping
//! - Renaming columns
//! - Filtering rows
//! - Filling null values
//! - Normalizing numeric columns
//! - Sorting and sampling
//! - Chaining multiple transforms
//!
//! Run with: cargo run --example transforms_pipeline

use std::sync::Arc;

use alimentar::{
    ArrowDataset, Cast, Chain, Dataset, Drop, FillNull, FillStrategy, Filter, NormMethod,
    Normalize, Rename, Select, Skip, Sort, SortOrder, Take, Transform, Unique,
};
use arrow::{
    array::{BooleanArray, Float64Array, Int32Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

fn create_sales_dataset() -> alimentar::Result<ArrowDataset> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int32, false),
        Field::new("product", DataType::Utf8, false),
        Field::new("quantity", DataType::Int32, false),
        Field::new("price", DataType::Float64, true), // nullable
        Field::new("discount", DataType::Float64, true), // nullable
        Field::new("region", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])),
            Arc::new(StringArray::from(vec![
                "Widget", "Gadget", "Widget", "Gizmo", "Gadget", "Widget", "Gizmo", "Gadget",
                "Widget", "Gizmo",
            ])),
            Arc::new(Int32Array::from(vec![10, 5, 3, 8, 12, 6, 15, 4, 9, 7])),
            Arc::new(Float64Array::from(vec![
                Some(29.99),
                Some(49.99),
                None, // missing price
                Some(19.99),
                Some(49.99),
                Some(29.99),
                None, // missing price
                Some(49.99),
                Some(29.99),
                Some(19.99),
            ])),
            Arc::new(Float64Array::from(vec![
                Some(0.1),
                None, // no discount
                Some(0.05),
                Some(0.15),
                None,
                Some(0.1),
                Some(0.2),
                None,
                Some(0.05),
                Some(0.1),
            ])),
            Arc::new(StringArray::from(vec![
                "North", "South", "East", "West", "North", "South", "East", "West", "North",
                "South",
            ])),
        ],
    )?;

    ArrowDataset::from_batch(batch)
}

fn main() -> alimentar::Result<()> {
    println!("=== Alimentar Transform Pipeline Example ===\n");

    let dataset = create_sales_dataset()?;

    println!(
        "Original dataset: {} rows, {} columns",
        dataset.len(),
        dataset.schema().fields().len()
    );
    println!(
        "Columns: {:?}",
        dataset
            .schema()
            .fields()
            .iter()
            .map(|f| f.name())
            .collect::<Vec<_>>()
    );

    // 1. Select specific columns
    println!("\n1. Select columns (order_id, product, quantity)");
    let select = Select::new(vec!["order_id", "product", "quantity"]);
    let selected = dataset.with_transform(&select)?;

    println!(
        "   Result: {} columns - {:?}",
        selected.schema().fields().len(),
        selected
            .schema()
            .fields()
            .iter()
            .map(|f| f.name())
            .collect::<Vec<_>>()
    );

    // 2. Drop columns
    println!("\n2. Drop column (discount)");
    let drop_transform = Drop::new(vec!["discount"]);
    let dropped = dataset.with_transform(&drop_transform)?;

    println!(
        "   Result: {} columns - {:?}",
        dropped.schema().fields().len(),
        dropped
            .schema()
            .fields()
            .iter()
            .map(|f| f.name())
            .collect::<Vec<_>>()
    );

    // 3. Rename columns
    println!("\n3. Rename column (quantity -> qty)");
    let rename = Rename::from_pairs([("quantity", "qty")]);
    let renamed = dataset.with_transform(&rename)?;

    println!(
        "   Result: {:?}",
        renamed
            .schema()
            .fields()
            .iter()
            .map(|f| f.name())
            .collect::<Vec<_>>()
    );

    // 4. Filter rows with closure
    println!("\n4. Filter rows (quantity > 5)");
    let filter = Filter::new(|batch: &RecordBatch| -> alimentar::Result<BooleanArray> {
        let qty_col = batch
            .column_by_name("quantity")
            .ok_or_else(|| alimentar::Error::column_not_found("quantity"))?;
        let qty_array = qty_col
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| alimentar::Error::invalid_config("Expected Int32Array"))?;

        let mask: BooleanArray = qty_array.iter().map(|v| v.map(|x| x > 5)).collect();
        Ok(mask)
    });
    let filtered = dataset.with_transform(&filter)?;

    println!("   Result: {} rows (quantity > 5)", filtered.len());

    // 5. Fill null values with a constant
    println!("\n5. Fill null values in 'price' column with 0.0");
    let fill_price = FillNull::new("price", FillStrategy::Float(0.0));
    let filled = dataset.with_transform(&fill_price)?;

    println!("   Filled null prices with 0.0");
    if let Some(batch) = filled.get(0) {
        if let Some(price_col) = batch.column_by_name("price") {
            println!("   Null count after fill: {}", price_col.null_count());
        }
    }

    // 6. Fill with Zero strategy
    println!("\n6. Fill null values with Zero strategy");
    let fill_zero = FillNull::new("discount", FillStrategy::Zero);
    let filled_zero = dataset.with_transform(&fill_zero)?;

    println!("   Filled null discounts with zero");
    if let Some(batch) = filled_zero.get(0) {
        if let Some(discount_col) = batch.column_by_name("discount") {
            println!("   Null count after fill: {}", discount_col.null_count());
        }
    }

    // 7. Normalize numeric columns
    println!("\n7. Normalize 'quantity' column (min-max)");
    let normalize = Normalize::new(["quantity"], NormMethod::MinMax);
    let normalized = dataset.with_transform(&normalize)?;

    println!("   Normalized quantity to [0, 1] range");
    if let Some(batch) = normalized.get(0) {
        if let Some(qty_col) = batch.column_by_name("quantity") {
            if let Some(arr) = qty_col.as_any().downcast_ref::<Float64Array>() {
                let min = arr.iter().flatten().fold(f64::INFINITY, f64::min);
                let max = arr.iter().flatten().fold(f64::NEG_INFINITY, f64::max);
                println!("   Min: {:.2}, Max: {:.2}", min, max);
            }
        }
    }

    // 8. Sort by column
    println!("\n8. Sort by quantity descending");
    let sort = Sort::by("quantity").order(SortOrder::Descending);
    let sorted = dataset.with_transform(&sort)?;

    println!("   Sorted dataset by quantity (descending)");
    if let Some(batch) = sorted.get(0) {
        if let Some(qty_col) = batch.column_by_name("quantity") {
            if let Some(arr) = qty_col.as_any().downcast_ref::<Int32Array>() {
                let values: Vec<_> = arr.iter().take(5).map(|v| v.unwrap_or(0)).collect();
                println!("   First 5 quantities: {:?}", values);
            }
        }
    }

    // 9. Take and Skip
    println!("\n9. Take first 5 rows");
    let take = Take::new(5);
    let taken = dataset.with_transform(&take)?;
    println!("   Result: {} rows", taken.len());

    println!("\n   Skip first 3 rows");
    let skip = Skip::new(3);
    let skipped = dataset.with_transform(&skip)?;
    println!("   Result: {} rows", skipped.len());

    // 10. Unique values
    println!("\n10. Get unique products");
    let unique = Unique::by(["product"]);
    let unique_products = dataset.with_transform(&unique)?;

    println!("   Unique products: {} rows", unique_products.len());

    // 11. Cast column type
    println!("\n11. Cast quantity from Int32 to Float64");
    let cast = Cast::new(vec![("quantity", DataType::Float64)]);
    let casted = dataset.with_transform(&cast)?;

    println!("   Cast complete");
    if let Some(batch) = casted.get(0) {
        if let Some(qty_col) = batch.column_by_name("quantity") {
            println!("   New type: {:?}", qty_col.data_type());
        }
    }

    // 12. Chain multiple transforms
    println!("\n12. Chain transforms: Select -> Rename -> Sort");
    let chain = Chain::new()
        .then(Select::new(vec!["order_id", "product", "quantity"]))
        .then(Rename::from_pairs([("quantity", "qty")]))
        .then(Sort::by("qty"));

    let chained = dataset.with_transform(&chain)?;

    println!("   Chain applied: {} transforms", chain.len());
    println!(
        "   Result columns: {:?}",
        chained
            .schema()
            .fields()
            .iter()
            .map(|f| f.name())
            .collect::<Vec<_>>()
    );

    // 13. Direct batch transformation
    println!("\n13. Direct batch transformation");
    if let Some(batch) = dataset.get(0) {
        let select_transform = Select::new(vec!["product", "price"]);
        let transformed = select_transform.apply(batch)?;
        println!(
            "   Transformed batch: {} rows, {} columns",
            transformed.num_rows(),
            transformed.num_columns()
        );
    }

    // 14. Conditional pipeline
    println!("\n14. Building conditional pipeline");

    let schema = dataset.schema();
    let fields: Vec<_> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    println!("   Columns: {:?}", fields);

    let mut pipeline = Chain::new();

    // Add transforms based on conditions
    if fields.contains(&"price") {
        pipeline = pipeline.then(FillNull::new("price", FillStrategy::Float(0.0)));
        println!("   Added: FillNull for price");
    }

    if fields.contains(&"quantity") {
        pipeline = pipeline.then(Normalize::new(["quantity"], NormMethod::MinMax));
        println!("   Added: Normalize for quantity");
    }

    let result = dataset.with_transform(&pipeline)?;
    println!("   Pipeline result: {} rows", result.len());

    // 15. Summary
    println!("\n15. Transform summary");
    println!("   Transforms demonstrated:");
    println!("   - Select: Choose specific columns");
    println!("   - Drop: Remove columns");
    println!("   - Rename: Rename columns");
    println!("   - Filter: Filter rows by condition");
    println!("   - FillNull: Fill missing values");
    println!("   - Normalize: Scale numeric values");
    println!("   - Sort: Order by columns");
    println!("   - Take/Skip: Limit rows");
    println!("   - Unique: Remove duplicates");
    println!("   - Cast: Change column types");
    println!("   - Chain: Combine multiple transforms");

    println!("\n=== Example Complete ===");
    Ok(())
}

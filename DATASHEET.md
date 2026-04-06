# Datasheet for simular Datasets

## Dataset: California Cities TSP

### Motivation

**Purpose**: Benchmark dataset for traveling salesman problem (TSP) optimization algorithms.

**Creator**: simular development team

**Funding**: Open source project

### Composition

**Instances**: 25 California cities with real geographic coordinates

**Features per instance**:
- `name`: City name (string)
- `lat`: Latitude (float, degrees)
- `lon`: Longitude (float, degrees)

**Data format**: YAML

**Size**: ~2KB

**Sampling**: Hand-selected major California cities for diverse geographic distribution

### Collection Process

**Source**: Public geographic data (US Census, OpenStreetMap)

**Methodology**: Manual curation of 25 cities representing diverse California regions

**Timeframe**: December 2024

### Preprocessing

- Coordinates rounded to 4 decimal places
- Validated against multiple sources
- DVC tracked for version control

### Uses

**Intended use**: Benchmarking TSP optimization algorithms

**Not recommended**: Production routing applications (use actual road networks)

### Distribution

**License**: MIT (same as simular)

**Location**: `data/raw/california_cities.yaml`

### Maintenance

**Updates**: As needed for benchmark improvements

**Contact**: Via GitHub issues

---

## Dataset: Benchmark Metrics

### Composition

Three JSON files in `metrics/`:
- `tsp_benchmark.json`: TSP algorithm performance
- `orbit_benchmark.json`: Orbital mechanics simulation accuracy
- `monte_carlo_benchmark.json`: Monte Carlo estimation convergence

Each contains:
- Mean values with 95% confidence intervals
- Cohen's d effect sizes
- Sample sizes (n >= 100)

### License

MIT License


/// Mutation operators for NAS genomes.
#[derive(Debug, Clone, Copy)]
pub enum NasMutation {
    /// Add a random layer
    AddLayer,
    /// Remove a random layer
    RemoveLayer,
    /// Change layer type
    ChangeType,
    /// Modify layer parameters
    ModifyParams,
    /// Toggle layer active state
    ToggleActive,
}

/// Mutate a genome using the specified operator.
pub fn mutate_genome(
    genome: &mut NasGenome,
    mutation: NasMutation,
    space: &NasSearchSpace,
    rng: &mut impl Rng,
) {
    match mutation {
        NasMutation::AddLayer => mutate_add_layer(genome, space, rng),
        NasMutation::RemoveLayer => mutate_remove_layer(genome, space, rng),
        NasMutation::ChangeType => mutate_change_type(genome, space, rng),
        NasMutation::ModifyParams => mutate_modify_params(genome, space, rng),
        NasMutation::ToggleActive => mutate_toggle_active(genome, rng),
    }
}

fn mutate_add_layer(genome: &mut NasGenome, space: &NasSearchSpace, rng: &mut impl Rng) {
    if genome.len() >= space.max_layers {
        return;
    }
    let layer_type = space.layer_types[rng.random_range(0..space.layer_types.len())];
    let mut config = LayerConfig::new(layer_type);

    if matches!(layer_type, LayerType::Dense | LayerType::Lstm) {
        config.units = Some(rng.random_range(space.units_range.0..=space.units_range.1));
    }
    if !space.activations.is_empty() {
        config.activation =
            Some(space.activations[rng.random_range(0..space.activations.len())].clone());
    }

    let pos = rng.random_range(0..=genome.len());
    genome.layers_mut().insert(pos, config);
}

fn mutate_remove_layer(genome: &mut NasGenome, space: &NasSearchSpace, rng: &mut impl Rng) {
    if genome.len() > space.min_layers {
        let idx = rng.random_range(0..genome.len());
        genome.layers_mut().remove(idx);
    }
}

fn mutate_change_type(genome: &mut NasGenome, space: &NasSearchSpace, rng: &mut impl Rng) {
    if !genome.is_empty() && !space.layer_types.is_empty() {
        let idx = rng.random_range(0..genome.len());
        let new_type = space.layer_types[rng.random_range(0..space.layer_types.len())];
        genome.layers_mut()[idx].layer_type = new_type;
    }
}

fn mutate_modify_params(genome: &mut NasGenome, space: &NasSearchSpace, rng: &mut impl Rng) {
    if genome.is_empty() {
        return;
    }
    let idx = rng.random_range(0..genome.len());
    let layer = &mut genome.layers_mut()[idx];

    if let Some(units) = layer.units {
        let delta = rng.random_range(-32i64..=32);
        let new_units =
            (units as i64 + delta).clamp(space.units_range.0 as i64, space.units_range.1 as i64);
        layer.units = Some(new_units as usize);
    }

    if rng.random_bool(0.3) && !space.activations.is_empty() {
        layer.activation =
            Some(space.activations[rng.random_range(0..space.activations.len())].clone());
    }
}

fn mutate_toggle_active(genome: &mut NasGenome, rng: &mut impl Rng) {
    if !genome.is_empty() {
        let idx = rng.random_range(0..genome.len());
        let layer = &mut genome.layers_mut()[idx];
        layer.active = !layer.active;
    }
}

/// Crossover two parent genomes to produce offspring.
pub fn crossover_genomes(
    parent1: &NasGenome,
    parent2: &NasGenome,
    rng: &mut impl Rng,
) -> NasGenome {
    // Single-point crossover
    if parent1.is_empty() {
        return parent2.clone();
    }
    if parent2.is_empty() {
        return parent1.clone();
    }

    let cut1 = rng.random_range(0..=parent1.len());
    let cut2 = rng.random_range(0..=parent2.len());

    let mut child_layers = Vec::new();

    // Take first part from parent1
    child_layers.extend(parent1.layers()[..cut1].iter().cloned());

    // Take second part from parent2
    child_layers.extend(parent2.layers()[cut2..].iter().cloned());

    NasGenome::from_layers(child_layers)
}

/// Evaluate architecture complexity (proxy for compute cost).
#[must_use]
pub fn architecture_complexity(genome: &NasGenome) -> f64 {
    let mut complexity = 0.0;

    for layer in genome.layers() {
        if !layer.active {
            continue;
        }

        let base = match layer.layer_type {
            LayerType::Dense => 1.0,
            LayerType::Conv2d => 2.0,
            LayerType::Lstm => 4.0,
            LayerType::Attention => 3.0,
            LayerType::BatchNorm | LayerType::MaxPool2d | LayerType::AvgPool2d => 0.1,
            LayerType::Dropout | LayerType::Skip => 0.0,
        };

        let units_factor = layer.units.unwrap_or(64) as f64 / 64.0;
        complexity += base * units_factor;
    }

    complexity
}
